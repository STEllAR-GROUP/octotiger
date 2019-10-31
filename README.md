# Octo-Tiger
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
[![link](https://circleci.com/gh/STEllAR-GROUP/octotiger/tree/master.svg?style=svg)](https://circleci.com/gh/STEllAR-GROUP/octotiger/tree/master)

> Note for maintainers: The base Docker image used by CircleCI needs to be built
> and updated manually. Neither HPX or any of the other dependencies update
> automatically. Relevant files are under
> [`tools/docker/base_image`](tools/docker/base_image).

## Quick Reference

* **Where to get help**:

	IRC Channel `ste||ar` on [freenode.net](https://freenode.net/)

* **Where to file issues**:

	[Octo-Tiger Issue Tracker](https://github.com/STEllAR-GROUP/octotiger/issues)

* **Wiki pages**:

    [Octo-Tiger Wiki](https://github.com/STEllAR-GROUP/octotiger/wiki)

## Publications

* Thomas Heller, Bryce Adelstein Lelbach, Kevin A Huck, John Biddiscombe, Patricia Grubel, Alice E Koniges, Matthias Kretz, Dominic Marcello, David Pfander, Adrian Serio, Juhan Frank, Geoffrey C Clayton, Dirk Pflüger, David Eder, Hartmut Kaiser. “Harnessing Billions of Tasks for a Scalable Portable Hydrodynamic Simulation of the Merger of Two Stars.” The International Journal of High Performance Computing Applications, Feb. 2019 [Link](https://journals.sagepub.com/doi/10.1177/1094342018819744)
* David Pfander, Gregor Daiß, Dominic Marcello, Hartmut Kaiser, Dirk Pflüger, “Accelerating Octo-Tiger: Stellar Mergers on Intel Knights Landing with HPX”, DHPCC++ Conference 2018 hosted by IWOCL, St Catherine’s College, Oxford, May 14, 2018 [Link](https://dl.acm.org/citation.cfm?doid=3204919.3204938)
* Gregor Daiß, Parsa Amini, John Biddiscombe, Patrick Diehl, Juhan Frank, Kevin Huck, Hartmut Kaiser, Dominic Marcello, David Pfander, Dirk Pflüger. “From Piz Daint to the Stars: Simulation of Stellar Mergers using High-Level Abstractions.”, Supercomputing Conference 2019, Nov. 2019 [Link (arXiv prepreint)](https://arxiv.org/abs/1908.03121) DOI: 10.1145/3295500.3356221

## License
Distributed under the Boost Software License, Version 1.0. (See 
<http://www.boost.org/LICENSE_1_0.txt>)
