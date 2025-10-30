import os
import time
import pylab as pl
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution

def simpleMD(init_conf, temp, calc, fname, s, T):
    """
    Uruchamia prostą symulację dynamiki molekularnej z termostatem Langevina.

    Argumenty:
        init_conf (ase.Atoms): Struktura początkowa.
        temp (float): Temperatura symulacji w Kelwinach.
        calc (ase.calculators.calculator.Calculator): Kalkulator ASE do obliczania energii i sił.
        fname (str): Nazwa pliku wyjściowego dla trajektorii (.xyz).
        s (int): Interwał zapisu klatek (co ile kroków zapisywać).
        T (int): Całkowita liczba kroków symulacji.
    """
    init_conf.calc = calc

    # Inicjalizacja prędkości i temperatury
    MaxwellBoltzmannDistribution(init_conf, temperature_K=temp)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = Langevin(init_conf, 1.0 * units.fs, temperature_K=temp, friction=0.02)

    # Przygotowanie do rysowania "na żywo"
    pl.ion()
    fig, ax = pl.subplots(2, 1, figsize=(8, 6), sharex='all', gridspec_kw={'hspace': 0})
    
    time_fs, temperature, energies = [], [], []
    
    if os.path.exists(fname):
        os.remove(fname)

    def write_frame():
        """Funkcja wywoływana co `s` kroków w celu zapisu danych i aktualizacji wykresu."""
        time_fs.append(dyn.get_time() / units.fs)
        temperature.append(init_conf.get_temperature())
        energies.append(init_conf.get_potential_energy() / len(init_conf))

        dyn.atoms.write(fname, append=True)
        
        # Aktualizacja wykresów
        ax[0].clear()
        ax[0].plot(time_fs, energies, color="b")
        ax[0].set_ylabel('E (eV/atom)')
        ax[0].grid(True)

        ax[1].clear()
        ax[1].plot(time_fs, temperature, color="r")
        ax[1].axhline(temp, linestyle='--', color='gray', label=f'Target T={temp}K')
        ax[1].set_ylabel('T (K)')
        ax[1].set_xlabel('Time (fs)')
        ax[1].grid(True)
        ax[1].legend()

        fig.canvas.draw()
        fig.canvas.flush_events()
        pl.pause(0.001)

    dyn.attach(write_frame, interval=s)
    
    print(f"Uruchamianie symulacji MD na {T} kroków...")
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    print(f"MD zakończone w {(t1 - t0) / 60:.2f} minut!")
    pl.ioff()
    pl.show()
