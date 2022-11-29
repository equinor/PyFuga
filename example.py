from pyfuga._fuga import get_luts


def main():
    if __name__ == '__main__':

        luts = get_luts(folder='.', zeta0=0, nkz0=16, nbeta=32,
                        diameter=80, zhub=70, z0=0.00001, zi=400, zlow=70, zhigh=70,
                        lut_vars=['UL'])

        import matplotlib.pyplot as plt
        luts.UL.sel(z=70)[502:542, :40].plot(x='x')
        plt.axis('equal')
        plt.show()


main()
