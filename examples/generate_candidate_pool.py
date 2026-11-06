"""Generate and inspect a finite Sobol candidate pool."""

from tra_bo import generate_sobol_pool


latent, configurations = generate_sobol_pool("patchtst", count=256, seed=17)
print("latent shape:", latent.shape)
print("first configuration:", configurations[0])
