package cmd

import (
	"context"
	"log"
	"os"
	ossignal "os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	signal "github.com/cmriat/webrtcssvr/server"
)

var (
	listenAddr     string
	originPatterns []string
)

// serveCmd starts the signaling server.
var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the WebRTC signaling server",
	RunE: func(cmd *cobra.Command, args []string) error {
		logger := log.New(os.Stdout, "webrtcssvr ", log.LstdFlags|log.Lmsgprefix)
		srv := signal.New(logger, signal.Options{
			Address:        listenAddr,
			OriginPatterns: originPatterns,
			ReadTimeout:    10 * time.Second,
			WriteTimeout:   10 * time.Second,
			IdleTimeout:    60 * time.Second,
		})

		// Context canceled on SIGINT/SIGTERM
		ctx, cancel := signalContext()
		defer cancel()
		return srv.Serve(ctx)
	},
}

func init() {
	rootCmd.AddCommand(serveCmd)
	serveCmd.Flags().StringVar(&listenAddr, "addr", ":8080", "address to listen on (host:port)")
	serveCmd.Flags().StringSliceVar(&originPatterns, "origins", []string{"*"}, "allowed Origin patterns for websockets")
}

func signalContext() (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan os.Signal, 1)
	ossignal.Notify(ch, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-ch
		cancel()
	}()
	return ctx, cancel
}
