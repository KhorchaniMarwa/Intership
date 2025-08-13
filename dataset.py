import rebound
import numpy as np
import pandas as pd

# ---- Settings ----
N_SAMPLES = 100       # Number of asteroids to simulate
INTEGRATION_TIME = 1e5  # years (set lower for quick test)
ORBITAL_A_RANGE = (2.0, 4.0)  # AU
ORBITAL_E_RANGE = (0.0, 0.4)
ORBITAL_I_RANGE = (0.0, 20.0)  # degrees

# ---- Setup base simulation ----
sim_base = rebound.Simulation()
sim_base.add(m=1.0)  # Sun
sim_base.add(m=0.0009543, a=5.2)   # Jupiter
sim_base.add(m=0.0002857, a=9.5)   # Saturn
sim_base.add(m=0.00004365, a=19.2) # Uranus
sim_base.add(m=0.00005149, a=30.1) # Neptune
sim_base.move_to_com()

# ---- Main loop ----
results = []
for i in range(N_SAMPLES):
    # Randomize asteroid's orbital elements
    a = np.random.uniform(*ORBITAL_A_RANGE)
    e = np.random.uniform(*ORBITAL_E_RANGE)
    inc = np.random.uniform(*ORBITAL_I_RANGE)
    omega = np.random.uniform(0, 360)
    Omega = np.random.uniform(0, 360)
    M = np.random.uniform(0, 360)

    # Start from base simulation
    sim = sim_base.copy()
    sim.add(m=0, a=a, e=e, inc=inc, omega=omega, Omega=Omega, M=M)

    # Integrate
    try:
        sim.integrate(sim.t + INTEGRATION_TIME, exact_finish_time=0)
        asteroid = sim.particles[-1]
        # Check if the asteroid is still in the system (not ejected)
        is_stable = int(abs(asteroid.a) < 100 and abs(asteroid.e) < 0.99)
    except rebound.Escape:
        # Asteroid was ejected
        is_stable = 0
    except Exception as ex:
        print(f"Error in sample {i}: {ex}")
        is_stable = 0

    results.append({
        "a": a,
        "e": e,
        "i": inc,
        "is_stable": is_stable
    })
    if (i+1) % 10 == 0:
        print(f"Simulated {i+1}/{N_SAMPLES} samples")

# ---- Save to CSV ----
df = pd.DataFrame(results)
df.to_csv("stability_dataset.csv", index=False)
print("Dataset saved to stability_dataset.csv")