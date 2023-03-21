import numpy as np
import matplotlib.pyplot as plt



def main():

    arr = np.loadtxt("data/SI_IPCS.npy")

    burn_in = arr.shape[0] // 2
    burn_in = arr.shape[0] // 5 * 4
    arr = arr[burn_in:,:]

    rho = 1.0
    mu = 1e-3
    U_m = 1.5
    U_bar = 2/3 * U_m
    D = 0.1
    Re = U_bar * D * rho / mu

    t = arr[:,0]
    C_D = 2 / (rho * U_bar**2 * D) * arr[:,1]
    C_L = 2 / (rho * U_bar**2 * D) * arr[:,2]
    p_diff = arr[:,3]
    C_D_max = np.amax(C_D)
    C_L_max = np.amax(C_L)
    p_mean = np.mean(p_diff)

    print(f"Max C_D = {np.amax(C_D):.2f}")
    print(f"Max C_L = {np.amax(C_L):.2f}")
    print(f"Max dP = {np.amax(p_diff):.2f}")

    """ Within ranges reported by group 8, most similar to our setup. """


    fig, axs = plt.subplots(1, 3)
    axs[0].plot(t, C_D, 'k-', label="Drag coefficient")
    axs[0].axhline(y=C_D_max, label=f"$y = {C_D_max:.2f}$", 
                   linestyle="--", color="black", alpha=0.8, lw=0.6)
    axs[0].legend()
    axs[0].set_title("Drag coefficient")
    axs[1].plot(t, C_L, 'k-', label="Lift coefficient")
    axs[1].axhline(y=C_L_max, label=f"$y = {C_L_max:.2f}$", 
                   linestyle="--", color="black", alpha=0.8, lw=0.6)
    axs[1].legend()
    axs[1].set_title("Lift coefficient")
    axs[2].plot(t, p_diff, 'k-', label="Pressure diff.")
    axs[2].axhline(y=p_mean, label=f"$y = {p_mean:.2f}$", 
                   linestyle="--", color="black", alpha=0.8, lw=0.6)
    axs[2].legend()
    axs[2].set_title("Pressure drop")



    plt.show()

    return


if __name__ == "__main__":
    main()
