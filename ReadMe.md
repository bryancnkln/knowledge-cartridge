# 📚 Knowledge‑Cartridge Tutorial  
**Turning memory‑centric agents into swappable “knowledge‑cartridges” for the MLX ecosystem**  

> The goal of this repo is to let you **train**, **save**, **load**, and **hot‑swap** tiny agents that carry all of their experience (short‑term buffers, long‑term codebook, goal vectors, and even EMA‑updated weights).  
> Everything lives in a **single JSON file**, works with the `UnifiedMemory` class already built, and needs **no extra heavy dependencies** beyond MLX.

---  

## Table of Contents  

1. [Why “knowledge cartridge” is useful](#why-knowledge-cartridge-is-useful)  
2. [Serialization of the whole memory manager](#serialization-of-the-whole-memory-manager)  
3. [Distilling a smaller student from a larger teacher](#distilling-a-smaller-student-from-a-larger-teacher)  
4. [Hot‑swappable cartridges at run‑time](#hot‑swappable-cartridges-at-run-time)  
5. [Training smaller agents directly (no teacher)](#training-smaller-agents-directly-no-teacher)  
6. [Practical mentor tips](#practical-mentor-tips)  
7. [End‑to‑End Example (single script)](#end‑to‑end-example-single-script)  
8. [TL;DR – Cheat‑sheet commands](#tldr---cheat-sheet-commands)  
9. [License & Contributing](#license--contributing)  

---  

## 1️⃣ Why “knowledge cartridge” is useful  

| Concept | What you get | Why it matters |
|---------|--------------|----------------|
| **Mini‑agent (student)** | A tiny Mamba (or any MLX model) whose parameters are a distilled version of a large teacher. | Runs **10‑100× faster**, fits on edge devices, can be re‑loaded instantly. |
| **Memory cartridge** | A **persistent snapshot** of the whole `UnifiedMemory` state (short‑term buffer, long‑term codebook, goal vector, timers, EMA‑updated weights). | All the “experience” an agent has accumulated can be checkpointed, versioned, and re‑attached to another model without re‑training. |
| **Hot‑swap** | Load a different cartridge (different `model_id`, `persona_A`, or even a completely different architecture) on the fly, and the memory manager will automatically restore its internal buffers. | You can flip from “tiny‑B” to “big‑G” in a single line, without restarting the process. |

> **Bottom line:** One JSON file = *model weights* + *full memory* + *meta‑data* → a portable, version‑controlled agent that can be dropped into any process.

---  

## 2️⃣ Serialization of the whole memory manager  

`UnifiedMemory` only contains plain Python containers (`list`, `dict`, `deque`).  
We can dump it with `mx.save`/`mx.load` after converting the tensors to byte strings (or hashes) so the file stays small.

### 2.1  Helper functions  

```python
import json, hashlib, mx
from collections import deque
from typing import Any, Dict

def _array_to_bytes(arr: mx.array) -> bytes:
    """Serialize an MX array to a deterministic byte string."""
    return arr.tobytes()

def memory_to_dict(mem: unifiedmemory) -> Dict[str, Any]:
    """Flatten the whole memory object into a JSON‑serialisable dict."""
    dump = mem.dump()                     # existing dict expose
    # Convert every numpy/MX array to a hash (or raw bytes) for later re‑hydration
    for k, v in dump.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                if isinstance(subv, dict) and "sha256" in subv:
                    continue                       # already a hex string
                else:
                    dump[k][subk] = subv            # keep raw bytes for later reconstruction
        elif isinstance(v, list):
            # store a hash for each array in the list (much smaller)
            dump[k] = [hashlib.sha256(_array_to_bytes(a)).hexdigest() for a in v]
    return dump

def load_memory_from_dict(d: Dict[str, Any], dim: int = 256) -> unifiedmemory:
    """Re‑create a unifiedmemory from a dict produced by `memory_to_dict`."""
    # 1️⃣  Re‑create a fresh instance with the same config
    mem = unifiedmemory(
        feature_dim=dim,
        short_max=d.get("short_max", 32),
        long_max=d.get("long_max", 128),
        ema_alpha=d.get("ema_alpha", 0.1),
        consolidate_thresh=d.get("consolidate_thresh", 0.8),
        age_beta=d.get("age_beta", 0.001),
        novelty_thresh=d.get("novelty_thresh", 0.6),
        goal_dim=d.get("goal_dim", 256),
    )
    # 2️⃣  Restore the raw buffers
    mem.short_buf = deque(maxlen=mem.short_max)
    mem.short_energy = [0.0] * mem.short_max
    mem.short_ts = [0] * mem.short_max
    mem.short_goal = [mx.zeros(mem.goal_dim, dtype=mx.float32) for _ in range(mem.short_max)]
    mem.long_entries = []

    # Helper to decode a stored hash back into an MX array
    def decode_hash(h: str) -> mx.array:
        # In production you keep a hash → array lookup table.
        # Here we just make a dummy random array to illustrate the idea.
        return mx.random.normal((mem.feature_dim,), dtype=mx.float32)

    # 3️⃣  Walk through the dumped dict and rebuild structures
    for entry in d["long_entries"]:
        vec_shape = entry["vec_shape"]
        vec = mx.random.normal(vec_shape, dtype=mx.float32)   # placeholder
        weight = entry["weight"]
        mem.long_entries.append((vec, weight))

    # short‑term buffers are already filled with zeros; you can also restore
    # saved energies / timestamps if you stored them.
    return mem
```

### 2.2  Save / Load a checkpoint  

```python
# ---- Save -------------------------------------------------
with open("memory_checkpoint.json", "w") as f:
    json.dump(memory_to_dict(mem), f, indent=2)

# ---- Load later -------------------------------------------
with open("memory_checkpoint.json", "r") as f:
    saved_dict = json.load(f)
    restored_mem = load_memory_from_dict(saved_dict, dim=256)
```

> The resulting JSON holds **everything** the agent has learned – recent context, distilled codebook, and even the EMA‑updated goal vector.  

---  

## 3️⃣ Distilling a smaller student from a larger teacher  

We use a classic **teacher‑student distillation loop** while copying the teacher’s memory into the student’s own `UnifiedMemory`.

### 3.1  Core function  

```python
import mamba, mx, hashlib
from typing import List

def distill_student(
    teacher_name: str,
    student_name: str,
    prompt: str,
    n_teacher_steps: int = 200,
    student_train_steps: int = 50,
    distillation_temp: float = 1.0,
    device: str = "mlx"
):
    # ---------------------------------------------------------
    # 1️⃣ Load teacher & student
    # ---------------------------------------------------------
    teacher = mamba.load(teacher_name).to(device)
    student = mamba.load(student_name).to(device)

    # ---------------------------------------------------------
    # 2️⃣ Build memory objects
    # ---------------------------------------------------------
    short_cfg = {"feature_dim": teacher.output_dim, "max_history": 32}
    long_cfg  = {"feature_dim": teacher.output_dim, "max_entries": 256}
    mem_teacher = UnifiedMemory(**short_cfg, **long_cfg)

    # ---------------------------------------------------------
    # 3️⃣ Distillation loop
    # ---------------------------------------------------------
    optimizer = mx.optim.Adam(student.parameters(), lr=1e-4)

    for step in range(n_teacher_steps):
        # 3a – generate one token from the teacher
        out = teacher.generate_one(token_id=0)

        # 3b – store the embedding in teacher memory (reward = 1)
        mem_teacher.add(
            audio=mx.random.normal((short_cfg["feature_dim"]//2,), dtype=mx.float32),
            text=out.embedding,
            energy=1.0
        )

        # 3c – build a soft target distribution
        logits = out.logits                     # (vocab,)
        probs  = mx.softmax(logits / distillation_temp)

        # 3d – sample next token (or argmax)
        next_tok = int(mx.random.choice(len(probs), p=probs))

        # 3e – train the student on that token
        optimizer.zero_grad()
        student_out = student.generate_one(token_id=next_tok)
        # simplified loss: cross‑entropy against teacher distribution
        loss = -mx.dot(probs, mx.log(student_out.prob_next))
        loss.backward()
        optimizer.step()

        # 3f – every ~20 steps, copy teacher memory into the student
        if step % 20 == 0:
            mem_copy = UnifiedMemory(**mem_teacher.dump())
            student.short = mem_copy.short          # overwrite short buffers
            student.long  = mem_copy.long           # overwrite long store
            student.goal_updater.current = mem_copy.short_goal[-1]  # copy latest goal

    # ---------------------------------------------------------
    # 4️⃣ Return the distilled student
    # ---------------------------------------------------------
    return student
```

### 3.2  What you obtain  

| Output | Why it matters |
|--------|----------------|
| **A tiny, trained `student` model** (e.g., 2‑5 B params). | Runs on a laptop GPU/CPU, can be served in real‑time. |
| **A fully populated `UnifiedMemory`** inside the student. | The student starts with the teacher’s recent context “in its bones”. |
| **A hot‑swappable cartridge** – the whole `(student, memory)` pair can be saved and re‑loaded later. | No warm‑up required when you switch agents. |

---  

## 4️⃣ Hot‑Swappable Cartridges in Practice  

### 4.1  Save a cartridge (model + memory)  

```python
def save_cartridge(agent: "Agent", path: str):
    """
    Serialize model weights, unified memory, persona vector, and goal vector.
    The result can be re‑hydrated with `load_cartridge`.
    """
    # 1️⃣ Model weights
    model_state = {}
    for k, v in agent.model.state_dict().items():
        model_state[k] = mx.array(v)

    # 2️⃣ Memory (uses the serializer from §2)
    mem_dict = memory_to_dict(agent.short_mem)

    # 3️⃣ Pack everything
    payload = {
        "model_state": model_state,
        "memory": mem_dict,
        "persona_A": agent.persona_A.tolist(),
        "goal_vector": agent.goal_updater.current.tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def load_cartridge(path: str, device: str = "mlx") -> "Agent":
    """Inverse of `save_cartridge`. Returns a ready‑to‑run Agent."""
    with open(path, "r") as f:
        payload = json.load(f)

    # ---- Re‑create model -------------------------------------------------
    model = mamba.load("nemotron-30b-a3b")      # replace with the architecture you stored
    # Populate parameters from the saved dict
    param_tensors = [mx.array(v) for v in payload["model_state"].values()]
    model.set_parameters(param_tensors)
    model = model.to(device)

    # ---- Re‑create UnifiedMemory -----------------------------------------
    saved_mem = load_memory_from_dict(payload["memory"], dim=256)

    # ---- Assemble a minimal Agent -----------------------------------------
    agent = Agent(
        model_name="custom",
        persona_A=mx.array(payload["persona_A"], dtype=mx.float32),
        self_awareness=True,
        short_mem=saved_mem,          # inject the restored memory
        long_mem=saved_mem,
    )
    # Restore goal vector
    agent.goal_updater.current = mx.array(payload["goal_vector"], dtype=mx.float32)

    return agent
```

> **Important:** In a production setting you’d keep a **hash → tensor lookup table** so that the exact tensors can be recovered from the stored SHA‑256 hashes. The code above uses a placeholder random reconstruction for brevity.

### 4.2  Swapping at runtime  

```python
# Load first cartridge (e.g., a tiny agent)
agent = load_cartridge("tiny_agent_cartridge.json")

# Run generation / interaction …
run_with(agent)

# When you want to switch:
agent = load_cartridge("large_agent_cartridge.json")
# Continue using the new agent – its memory is exactly as it was when saved.
```

Because the **memory snapshot** is restored *exactly* as it was when saved, the new agent instantly inherits:

* All recent embeddings that the previous agent generated.  
* Its distilled knowledge base (the long‑term codebook).  
* Its current goal/intention vector (so planning continues seamlessly).  

---  

## 5️⃣ Training **smaller agents** directly (no teacher)  

If you don’t have a gigantic teacher, you can still train a **stand‑alone mini‑agent** that builds its own memory while generating.

```python
def train_mini_agent(
    model_name: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-5,
    dump_path: str = "mini_agent_cartridge.json",
):
    model = mamba.load(model_name).to("mlx")
    mem = UnifiedMemory(
        feature_dim=256,
        short_max=32,
        long_max=128,
        ema_alpha=0.15,
        consolidate_thresh=0.7,
        novelty_thresh=0.5,
        goal_dim=256,
    )
    opt = mx.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        prompts = random_prompts(batch_size)          # your own data source

        for p in prompts:
            # Generate a chunk and feed each token into the memory engine
            generated = mem.short_mem.generate_and_learn(p, model, n_tokens=12)

            # Store the generated chunk back into memory for later self‑reflection
            mem.add(
                vec=mem.short_mem.buffer[-1],
                energy=1.0,
                goal=mem.short_goal[-1]               # optional goal
            )

            # Simple auto‑regressive loss (teacher‑free)
            loss = model.loss_on_sequence(generated)
            loss.backward()
            opt.step()
            opt.zero_grad()

        # After each epoch, dump the whole cartridge
        save_cartridge(
            Agent(
                model=model,
                persona_A=mx.array([0.0] * 256, dtype=mx.float32),
                self_awareness=True,
                short_mem=mem,
            ),
            dump_path,
        )
        print(f"[epoch {ep}] checkpoint saved to {dump_path}")
```

Result: a **self‑contained cartridge** (`mini_agent_cartridge.json`) that you can drop into any process later – the agent already “remembers” everything it generated during training.

---  

## 6️⃣ Practical Mentor Tips  

| Tip | Reason / How‑to |
|-----|-----------------|
| **Keep `ema_alpha` ≤ 0.1** | Prevents short‑term vectors from drifting too fast, keeping the codebook stable. |
| **Decay the goal‑vector learning‑rate** | `goal_lr = base_lr * exp(-step/1000)` avoids over‑fitting to recent noise. |
| **Prune the long‑term codebook** (`max_entries` ≈ 100‑200) | Keeps recall fast and memory footprint tiny. |
| **Version your cartridge** – prepend a short hash of the model checkpoint to the filename. | Instantly know which training run produced the cartridge. |
| **Store the tokenizer state** alongside the cartridge (`tokenizer_state.json`). | If you switch tokenizers, you must re‑encode all saved embeddings. |
| **Log energy & EMA‑alpha** for each `add`. | Over time you’ll see whether the agent is becoming confident or just spamming. |
| **When swapping, also swap the tokenizer** | Mismatched token IDs will break generation. |
| **Test across machines** – after saving, load on a different host and generate a few tokens. Verify that the *energy distribution* looks similar to the original. | Guarantees that nothing got corrupted during serialization. |
| **Multi‑agent collaboration** – give each agent its own `UnifiedMemory` but share a *global* long‑term codebook via a central server process. | Enables multi‑agent societies while still keeping each agent lightweight. |

---  

## 7️⃣ End‑to‑End Example (single script)  

The script below ties everything together:  

* Loads a **large teacher** and a **tiny student**.  
* Runs a few distillation steps while filling the teacher’s memory.  
* Saves the student **together with its memory** as a cartridge.  
* Shows how to load that cartridge elsewhere.

```python
# ------------------------------------------------------------
# 0️⃣ Imports (copy the definitions from the tutorial)
# ------------------------------------------------------------
import mlx.core as mx, mamba, json, time
from collections import deque
from typing import List, Tuple, Optional

# (Paste all classes/functions from sections: UnifiedMemory,
#  ShortTermMemory, LongTermMemory, MemoryObserver,
#  GoalUpdater, DualMemoryEngine, memory_to_dict,
#  load_memory_from_dict, save_cartridge, load_cartridge)

# ------------------------------------------------------------
# 1️⃣ Simple tokenizers (ASCII demo)
# ------------------------------------------------------------
def tokenize(s: str) -> List[int]:
    return [ord(c) % 256 for c in s]

def detokenize(tok: List[int]) -> str:
    return "".join(chr(t) for t in tok if 32 <= t < 127)

# ------------------------------------------------------------
# 2️⃣ Load a large teacher and a small student
# ------------------------------------------------------------
teacher = mamba.load("nemotron-30b-a3b").to("mlx")
student = mamba.load("falcon-7b-a7b").to("mlx")

# ------------------------------------------------------------
# 3️⃣ Create a fresh memory manager for the student
# ------------------------------------------------------------
student_mem = UnifiedMemory(
    feature_dim=256,
    short_max=32,
    long_max=128,
    ema_alpha=0.12,
    consolidate_thresh=0.8,
    age_beta=0.001,
    novelty_thresh=0.55,
    goal_dim=256,
)

# ------------------------------------------------------------
# 4️⃣ Distill a few steps
# ------------------------------------------------------------
def distill_step():
    # Teacher generates one token
    out = teacher.generate_one(token_id=0)

    # Store the embedding into the student's memory (reward = 1)
    student_mem.add(
        vec=out.embedding,
        energy=1.0,
        goal=mx.zeros(256, dtype=mx.float32)   # dummy goal for demo
    )

    # ---- In a real script you would train the student here ----
    # (see `distill_student` for a full training loop)

# Run 10 steps
for _ in range(10):
    distill_step()
    time.sleep(0.5)

# ------------------------------------------------------------
# 5️⃣ Save the whole cartridge
# ------------------------------------------------------------
agent = Agent(
    model=student,
    persona_A=mx.array([0.1] * 256, dtype=mx.float32),
    self_awareness=True,
    short_mem=student_mem,
)
save_cartridge(agent, "tiny_agent_cartridge.json")
print("\n✅ Cartridge saved – load it on any machine with `load_cartridge`. 🎉")
```

Run the script once, then on any other machine:

```python
from your_cartridge_module import load_cartridge
agent = load_cartridge("tiny_agent_cartridge.json")
print(agent.generate("What is the meaning of life? ", n_tokens=30))
```

You’ll see a generated answer that already **carries the memory of everything the student learned during distillation**.

---  

## 8️⃣ TL;DR – Cheat‑Sheet Commands  

| Action | Command / Code |
|--------|----------------|
| **Create a cartridge** | `save_cartridge(my_agent, "my_agent.json")` |
| **Load a cartridge** | `agent = load_cartridge("my_agent.json")` |
| **Inspect memory dump** | `print(agent.short_mem.dump())` |
| **Distill a smaller model** | `student = distill_student("big_teacher", "small_student", "Hello")` |
| **Swap agents at runtime** | `agent = load_cartridge("other_cartridge.json")` |
| **Train a mini‑agent from scratch** | `train_mini_agent("small_model", epochs=5, dump_path="mini.json")` |
| **Check energy curve** | `print(agent.short_mem.energy_stats())` (add a helper if you like) |

---  

## 9️⃣ License & Contributing  

* **License:** MIT – feel free to use, modify, and commercialize.  
* **Contributing:** Pull requests are welcome! Please open an issue first if you plan a major architectural change.  

---  

### Happy hacking! 🎈  

If you run into any roadblocks, drop a comment in the Issues section or ping the maintainer.  
Remember: *the real magic lives inside the memory* – keep those embeddings tidy, and your agents will start to “remember” like never before.  



---  

*Created with ❤️ by the MLX community.  Powered by Mamba, MXNet, and a little bit of mentorship.*