from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex

PATTERNS_TO_REPLACE = [
    "_orig_mod.",  # this pattern must replaced
    "._orig_mod",
    "._fsdp_wrapped_module",
    "._checkpoint_wrapped_module",
    ".module",
    "_module.",
]

def normalize_state_dict_key(key: str, patterns_to_remove=PATTERNS_TO_REPLACE) -> str:
    out = key
    for p in patterns_to_remove:
        out = out.replace(p, "")
    return out


class MetadataNormalizingPlanner(DefaultLoadPlanner):
    def set_up_planner(self, state_dict, metadata: Metadata | None = None, is_coordinator: bool = False) -> None:
        if metadata is not None:
            # 1.1 state_dict_metadata: Dict[str, ...]
            if hasattr(metadata, "state_dict_metadata") and isinstance(metadata.state_dict_metadata, dict):
                old = metadata.state_dict_metadata
                new = {}
                collisions = []
                for k, v in old.items():
                    nk = normalize_state_dict_key(k)
                    if nk in new and nk != k:
                        collisions.append((k, nk))
                    new[nk] = v
                old.clear()
                old.update(new)

                if collisions and is_coordinator:
                    print("[warn] state_dict_metadata collisions after normalization (first 10):")
                    for src, dst in collisions[:10]:
                        print(f"  {src} -> {dst}")

            # 1.2 storage_data: Dict[MetadataIndex, StorageInfo]
            if hasattr(metadata, "storage_data") and isinstance(metadata.storage_data, dict):
                old_sd = metadata.storage_data
                new_sd = {}
                collisions = []
                for idx, info in old_sd.items():
                    # idx: MetadataIndex(fqn=..., offset=..., index=...)
                    new_idx = MetadataIndex(
                        fqn=normalize_state_dict_key(idx.fqn),
                        offset=idx.offset,
                        index=idx.index,
                    )
                    if new_idx in new_sd and new_idx != idx:
                        collisions.append((idx, new_idx))
                    new_sd[new_idx] = info
                old_sd.clear()
                old_sd.update(new_sd)

                if collisions and is_coordinator:
                    print("[warn] storage_data collisions after normalization (first 10):")
                    for src, dst in collisions[:10]:
                        print(f"  {src} -> {dst}")
        super().set_up_planner(state_dict, metadata, is_coordinator)

