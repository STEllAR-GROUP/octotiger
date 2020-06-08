# Octo-Tiger [![link](https://circleci.com/gh/STEllAR-GROUP/octotiger/tree/master.svg?style=shield)](https://circleci.com/gh/STEllAR-GROUP/octotiger/tree/master)  [![Codacy Badge](https://app.codacy.com/project/badge/Grade/ebc6d3e2e4f0407aa6a80dfc4fd03b97)](https://www.codacy.com/gh/STEllAR-GROUP/octotiger?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=STEllAR-GROUP/octotiger&amp;utm_campaign=Badge_Grade)
```
          ___       _      _____ _                 
         / _ \  ___| |_ __|_   _(_) __ _  ___ _ __ 
        | | | |/ __| __/ _ \| | | |/ _` |/ _ \ '__|
        | |_| | (__| || (_) | | | | (_| |  __/ |   
         \___/ \___|\__\___/|_| |_|\__, |\___|_|   
      _                            |___/           
     (_)
              _ __..-;''`-
      O   (`/' ` |  \ \ \ \-.
       o /'`\ \   |  \ | \|  \
        /<7' ;  \ \  | ; ||/ `'.
       /  _.-, `,-\,__| ' ' . `'.
       `-`  f/ ;       \        \             ___.--,
           `~-'_.._     |  -~    |     _.---'`__.-( (_.
        __.--'`_.. '.__.\    '--. \_.-' ,.--'`     `""`
       ( ,.--'`   ',__ /./;   ;, '.__.'`    __
       _`) )  .---.__.' / |   |\   \__..--""  """--.,_
      `---' .'.''-._.-'`_./  /\ '.  \ _.-~~~````~~~-._`-.__.'
            | |  .' _.-' |  |  \  \  '.               `~---`
             \ \/ .'     \  \   '. '-._)
              \/ /        \  \    `=.__`~-.
              / /\         `) )    / / `"".`\
        , _.-'.'\ \        / /    ( (     / /
         `--~`   ) )    .-'.'      '.'.  | (
                (/`    ( (`          ) )  '-;
                 `      '-;         (-'
```

From <https://doi.org/10.1145/3204919.3204938>:
> Octo-Tiger is an astrophysics program simulating the evolution of star systems
> based on the fast multipole method on adaptive Octrees. It was implemented using
> high-level C++ libraries, specifically HPX and Vc, which allows its use on
> different hardware platforms

## Build Status

Current status of the [CircleCI](https://circleci.com/gh/STEllAR-GROUP/octotiger) continuous
integration service for the master branch:
[![link](https://circleci.com/gh/STEllAR-GROUP/octotiger/tree/master.svg?style=shield)](https://circleci.com/gh/STEllAR-GROUP/octotiger/tree/master)

> Note for maintainers: The base Docker image used by CircleCI needs to be built
> and updated manually. Neither HPX or any of the other dependencies update
> automatically. Relevant files are under
> [`tools/docker/base_image`](tools/docker/base_image).

## Quick Reference

* **Where to get help**:

	IRC Channel `#ste||ar` on [freenode.net](https://freenode.net/)

* **Where to file issues**:

	[Octo-Tiger Issue Tracker](https://github.com/STEllAR-GROUP/octotiger/issues)

* **Wiki pages**:

    [Octo-Tiger Wiki](https://github.com/STEllAR-GROUP/octotiger/wiki)

## Publications

* Thomas Heller, Bryce Adelstein Lelbach, Kevin A Huck, John Biddiscombe, Patricia Grubel, Alice E Koniges, Matthias Kretz, Dominic Marcello, David Pfander, Adrian Serio, Juhan Frank, Geoffrey C Clayton, Dirk Pflüger, David Eder, Hartmut Kaiser. “Harnessing Billions of Tasks for a Scalable Portable Hydrodynamic Simulation of the Merger of Two Stars.” The International Journal of High Performance Computing Applications, Feb. 2019 [Link](https://journals.sagepub.com/doi/10.1177/1094342018819744)
* David Pfander, Gregor Daiß, Dominic Marcello, Hartmut Kaiser, Dirk Pflüger, “Accelerating Octo-Tiger: Stellar Mergers on Intel Knights Landing with HPX”, DHPCC++ Conference 2018 hosted by IWOCL, St Catherine’s College, Oxford, May 14, 2018 [Link](https://dl.acm.org/citation.cfm?doid=3204919.3204938)
* Gregor Daiß, Parsa Amini, John Biddiscombe, Patrick Diehl, Juhan Frank, Kevin Huck, Hartmut Kaiser, Dominic Marcello, David Pfander, and Dirk Pfüger. "From piz daint to the stars: simulation of stellar mergers using high-level abstractions." In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1-37. 2019. [Link](https://dl.acm.org/doi/abs/10.1145/3295500.3356221), [Pre-print](https://arxiv.org/abs/1908.03121)

## License
Distributed under the Boost Software License, Version 1.0. (See 
<http://www.boost.org/LICENSE_1_0.txt>)
