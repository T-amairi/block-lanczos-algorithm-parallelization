# Parallélisation de l'algorithme de Lanczos par bloc

Le but de ce projet est de paralléliser l’algorithme de Lanczos par bloc permettant
de résoudre les systèmes linéaires de la forme suivante : Ax = 0 mod p. Pour cela, on
implémentera différentes versions :

- MPI

- OpenMP

- hybride, i.e, MPI + OpenMP

Les matrices lues par le programme sont au format ```mtx```. Il est possible d'en récupérer [ici](https://suitesparse-collection-website.herokuapp.com/) ou bien via l'utilisation du script ```project.py```. Le sujet du projet et le rapport se trouve dans le dossier ```doc```.

## Lancement

Le makefile situé à la racine permet de compiler simultanément toutes les versions.

- Options :

```
--matrix FILENAME           MatrixMarket file containing the sparse matrix
--prime P                   compute modulo P
--n N                       blocking factor [default 1]
--output-file FILENAME      store the block of kernel vectors
--right                     compute right kernel vectors
--left                      compute left kernel vectors [default]
--stop-after N              stop the algorithm after N iterations
--checkpoint cp             enable checkpointing every cp seconds [default cp = 60 s]
--load-checkpoint           load vectors from checkpointing files

The --matrix and --prime arguments are required
The --stop-after and --output-file arguments mutually exclusive
```

- Commandes :

```bash
MPI :
$ mpiexec -machinefile hostfile --map-by ppr:1:core ./lanczos_modp [options]

OpenMP :
$ export OMP_STACKSIZE="1G" && ./lanczos_modp [options]

Hybrid :
$ export OMP_STACKSIZE="1G" && mpiexec -x OMP_STACKSIZE="1G" -machinefile hostfile --map-by ppr:1:node ./lanczos_modp [options]
```
