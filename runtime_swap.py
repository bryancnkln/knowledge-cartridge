# 1️⃣ Load the first cartridge (e.g., a “tiny” agent)
# ------------------------------------------------------------
cart1 = load_cartridge("tiny_agent_cartridge.json")
run_with(cart1)          # your existing generation loop

# ------------------------------------------------------------
# 2️⃣  When you want to switch, just load another cartridge
# ------------------------------------------------------------
cart2 = load_cartridge("large_agent_cartridge.json")
switch_to(cart2)         # replace the global `agent` reference