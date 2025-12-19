# Save
with open("memory_checkpoint.json", "w") as f:
    json.dump(memory_to_dict(mem), f, indent=2)

# Load later
with open("memory_checkpoint.json", "r") as f:
    saved_dict = json.load(f)
    restored_mem = load_memory_from_dict(saved_dict, dim=256)