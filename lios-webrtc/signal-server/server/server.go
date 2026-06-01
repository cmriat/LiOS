package server

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

// Options configures the signaling server.
type Options struct {
	Address        string
	OriginPatterns []string
	ReadTimeout    time.Duration
	WriteTimeout   time.Duration
	IdleTimeout    time.Duration
}

// Server is a minimal WebRTC signaling server.
type Server struct {
	mu     sync.RWMutex
	rooms  map[string]map[string]*client // room -> peerID -> client
	logger *log.Logger
	opts   Options
}

// client represents a connected peer.
type client struct {
	id     string
	room   string
	conn   *websocket.Conn
	server *Server
	send   chan any
}

// Envelope is the generic message format exchanged over WebSocket.
// Common types: "join", "offer", "answer", "candidate", "leave".
// Server-emitted types: "peers", "peer-join", "peer-leave", "error".
type Envelope struct {
	Type string          `json:"type"`
	Room string          `json:"room,omitempty"`
	From string          `json:"from,omitempty"`
	To   string          `json:"to,omitempty"`
	Data json.RawMessage `json:"data,omitempty"`
	Text string          `json:"text,omitempty"`
}

// New creates a new Server instance.
func New(logger *log.Logger, opts Options) *Server {
	if logger == nil {
		logger = log.Default()
	}
	if opts.Address == "" {
		opts.Address = ":8080"
	}
	if opts.ReadTimeout == 0 {
		opts.ReadTimeout = 10 * time.Second
	}
	if opts.WriteTimeout == 0 {
		opts.WriteTimeout = 10 * time.Second
	}
	if opts.IdleTimeout == 0 {
		opts.IdleTimeout = 60 * time.Second
	}
	return &Server{
		rooms:  make(map[string]map[string]*client),
		logger: logger,
		opts:   opts,
	}
}

// Serve starts the HTTP server and blocks until it stops.
func (s *Server) Serve(ctx context.Context) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/ws", s.wsHandler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })

	srv := &http.Server{
		Addr:         s.opts.Address,
		Handler:      mux,
		ReadTimeout:  s.opts.ReadTimeout,
		WriteTimeout: s.opts.WriteTimeout,
		IdleTimeout:  s.opts.IdleTimeout,
	}

	// Shutdown when context is canceled.
	go func() {
		<-ctx.Done()
		_ = srv.Shutdown(context.Background())
	}()

	s.logger.Printf("listening on %s", s.opts.Address)
	err := srv.ListenAndServe()
	if err == http.ErrServerClosed {
		return nil
	}
	return err
}

func (s *Server) wsHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: s.opts.OriginPatterns})
	if err != nil {
		s.logger.Printf("accept error: %v", err)
		return
	}

	// Connection lifetime context with idle timeout pings.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	// Require an initial join message to register the client.
	var env Envelope
	if err := wsjson.Read(ctx, conn, &env); err != nil {
		_ = wsjson.Write(ctx, conn, Envelope{Type: "error", Text: "failed to read join: " + err.Error()})
		conn.Close(websocket.StatusProtocolError, "join required")
		return
	}
	if env.Type != "join" || env.Room == "" || env.From == "" {
		_ = wsjson.Write(ctx, conn, Envelope{Type: "error", Text: "invalid join: need type=join, room, from"})
		conn.Close(websocket.StatusProtocolError, "invalid join")
		return
	}

	c := &client{
		id:     env.From,
		room:   env.Room,
		conn:   conn,
		server: s,
		send:   make(chan any, 32),
	}

	if err := s.register(c); err != nil {
		_ = wsjson.Write(ctx, conn, Envelope{Type: "error", Text: err.Error()})
		conn.Close(websocket.StatusInternalError, "register failed")
		return
	}

	// Send current peers to the new client.
	peers := s.peersInRoom(c.room, c.id)
	_ = wsjson.Write(ctx, conn, Envelope{Type: "peers", Room: c.room, From: "server", Data: mustJSON(peers)})

	// Fan-out writer goroutine.
	go c.writer()

	// Read loop.
	for {
		var m Envelope
		if err := wsjson.Read(ctx, conn, &m); err != nil {
			// Normal closure/noise handling.
			status := websocket.CloseStatus(err)
			if status == websocket.StatusNormalClosure || status == websocket.StatusGoingAway || err == context.Canceled {
				break
			}
			s.logger.Printf("read error peer=%s room=%s: %v", c.id, c.room, err)
			break
		}

		// Default room/from if omitted after join.
		if m.Room == "" {
			m.Room = c.room
		}
		if m.From == "" {
			m.From = c.id
		}

		switch m.Type {
		case "leave":
			c.close(websocket.StatusNormalClosure, "leave")
			return
		case "offer", "answer", "candidate":
			// Forward to a specific peer in the same room.
			if m.To == "" {
				_ = wsjson.Write(ctx, conn, Envelope{Type: "error", Text: "missing 'to' field"})
				continue
			}
			s.forwardTo(m.Room, m.To, m)
		default:
			// Broadcast any other message types to the room (excluding sender).
			s.broadcast(m.Room, m.From, m)
		}
	}

	c.close(websocket.StatusNormalClosure, "disconnected")
}

func (s *Server) register(c *client) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	peers, ok := s.rooms[c.room]
	if !ok {
		peers = make(map[string]*client)
		s.rooms[c.room] = peers
	}
	if _, exists := peers[c.id]; exists {
		return &wsError{msg: "peer id already connected in this room"}
	}
	peers[c.id] = c
	// Notify others in the room.
	s.asyncBroadcastLocked(c.room, c.id, Envelope{Type: "peer-join", Room: c.room, From: c.id})
	s.logger.Printf("joined room=%s peer=%s", c.room, c.id)
	return nil
}

func (s *Server) unregister(c *client) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if peers, ok := s.rooms[c.room]; ok {
		if _, exists := peers[c.id]; exists {
			delete(peers, c.id)
			s.asyncBroadcastLocked(c.room, c.id, Envelope{Type: "peer-leave", Room: c.room, From: c.id})
			if len(peers) == 0 {
				delete(s.rooms, c.room)
			}
		}
	}
	s.logger.Printf("left room=%s peer=%s", c.room, c.id)
}

func (s *Server) peersInRoom(room, exclude string) []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []string
	if peers, ok := s.rooms[room]; ok {
		for id := range peers {
			if id == exclude {
				continue
			}
			out = append(out, id)
		}
	}
	return out
}

func (s *Server) forwardTo(room, to string, m Envelope) {
	s.mu.RLock()
	dst := (*client)(nil)
	if peers, ok := s.rooms[room]; ok {
		dst = peers[to]
	}
	s.mu.RUnlock()
	if dst != nil {
		select {
		case dst.send <- m:
		default:
			s.logger.Printf("drop message to=%s room=%s: slow consumer", to, room)
		}
	}
}

func (s *Server) broadcast(room, exclude string, m Envelope) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	s.asyncBroadcastLocked(room, exclude, m)
}

func (s *Server) asyncBroadcastLocked(room, exclude string, m Envelope) {
	if peers, ok := s.rooms[room]; ok {
		for id, p := range peers {
			if id == exclude {
				continue
			}
			select {
			case p.send <- m:
			default:
				s.logger.Printf("drop broadcast to=%s room=%s: slow consumer", id, room)
			}
		}
	}
}

func (c *client) writer() {
	// Periodic ping to keep connections alive.
	ticker := time.NewTicker(20 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case v, ok := <-c.send:
			if !ok {
				return
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			err := wsjson.Write(ctx, c.conn, v)
			cancel()
			if err != nil {
				return
			}
		case <-ticker.C:
			_ = c.conn.Ping(context.Background())
		}
	}
}

func (c *client) close(status websocket.StatusCode, reason string) {
	// Unregister first to prevent further sends to this client.
	c.server.unregister(c)
	close(c.send)
	_ = c.conn.Close(status, reason)
}

type wsError struct{ msg string }

func (e *wsError) Error() string { return e.msg }

func mustJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return json.RawMessage(b)
}
