# Epidemic simulation with CUDA and python

### Overview

This is a simulation of spread of an epidemic in a closed community. It utilises CUDA to compute every iteration of the simulation, and a python script to animate ready results, together with graphs depicting the health state of the population over time. The simulation is configurable under many aspects, which will be described bellow. The population is represented during visualization as points on a surface, with their color represting their health state in a following way
| Color | Health |
| ----- | ------ |
| Green | Healthy |
| Yellow | Sick without sympthoms |
| Red | Sick with sympthoms |
| Blue | Immune |

### Requirements
- CUDA (tested on 10.1)
- python (tested on 3.8)
- matplotlib
- numpy

### Startup

Compile pandemia.cu file

```sh
$ nvcc pandemia.cu -o pandemia
```

Run the simulation

```sh
$ ./pandemia.cu
```

You can specify startup settings like so (full set of options below)
```sh
$ ./pandemia.cu -N 10000 -DIM 100.0 --infection_p 0.1
```

Run the visualization script
```sh
$ python animate.py
```

### Simulation settings

Below is the list of every configurable setting. You can provide short or long version.
In the last column the type of an argument is given. For example, you would provide respectively
./pandemia -N 1000 -DIM 50.0

| Setting | Description | Short version | Long version |  Argument | Default value |
| ------ | ------ | ----- | ------ | ----- | ----- | 
| Poulation size | How many people to sumulate | -N | --N | int > 0 | 10000 |
| Dimention | Size of the simulated area (it will be a square of size DIMxDIM) | -DIM | --DIM | float > 0.0 | 100.0 |
| Simulation time | How many iterations to simulate | -simn | --simulation_n | int > 0 | 500 |
| Velocity | Distnace a person will move in a single iteration | -v | --velocity | float > 0.0 | 1.0 |
| Infection probability | Probability that a person within infection range of a sick individual will catch a disease | -infp | --infection_p | 0.0 < float < 1.0 | 0.33 |
| Infection range | How close has one to be to sick person to catch a disease | -infr | --infection_r | float > 0.0 | 3.0 |
| Immune time | Time a person will be sick before they'll become immune to the disease | -immt | --immune_time | int > 0 | 100 |
| Sympthoms time | Time a person will be sick before they'll have visible sympthoms | -symt | --sympthoms_time | int > 0 | 10 |
| Blocks | How many CUDA blocks to use | -b | --blocks | int > 0 | 128 |
| Threads per block | Threads per CUDA block | -tpb | --threads_per_block | int > 0 | 128 |
| Output | Name of an output file. Provide 'none' to not save output. (WARNING: currently animate.py is hardcoded to read from file 'output.sim' (I know, it's stupid)) | -o | --output | string | "output.sim" |
| Quarantine time for all | Time after which everyone will be subjected to quarantine. If set to 0, no quarantine for all will take place | -qat | --qarantine_all_time | int > 0 | 0 |
| Quarantine time for sick people | Time after which all people with visible sympthoms will be quarantined. If set to 0, no quarantine for sick people only will take place | -qst | --quarantine_sick_time | int > 0 | 0 |
| Lawful population | Percentage of population that will conform to quarantine rules | -lawp | --lawful_p | 0.0 < float < 1.0 | 1.0 |
| Number of gathering points | Number of points of interest (eg. markets) that people without sympthoms will visit during quarantine  | -gn | --gathering_points_n | int > 0 | 0 |
| Probability of visit to a gathering point | Probability that a person during quarantine will decide to go to the gathering place | -gp | --gathering_point_p | 0.0 < float < 1.0 | 0.05 |
| Buffor size | How many iterations to simulate before coping memory and writing it to file (It may optimalize simulation somehow) | -buff | --buffor_size | int > 0 | 1 |

### Things to improve that I can't be bothered with

 - Seperate code into diffrent files.
 - Make a filename in animate.py a providable argument, not a hardcoded one
 - Rewrite animation completly. Matplotlib doesn't do well with input size of over 50000. OpenGL would be preffered.
 - Try to utilize shared memory in CUDA. Maybe make it so that a CUDA thread does not manage people, but some part of a region.
 - Make a binary output file, not a text one. It would shrink it around twofold. Currently, with population size of 100000 and simulation time of 500 iterations it can go over 1GB.
 - Throw away a save to file completly (look at point 2).
 - Change some values like Immune time or valocity to bell curve, not a fixed probalility.
 - Write more complex rules for quarantine time that look at current state of population health.
 - Add more configurable parameters.
