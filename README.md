# Extremely Random Trees

## What

Extremely Random Trees is a machine-learning algorithm described in:

> "Extremely randomized trees", DOI 10.1007/s10994-006-6226-1,  
> by Pierre Geurts, Damien Ernst, Louis Wehenkel, 2005

This is my own implementation of that algorithm as a friendly terminal program.
It can be compiled and executed on any modern Unix/Linux/OSX computer.

## Building

I've provided a Makefile which should do the job.
After you've downloaded this repository, get into its directory in a terminal and do:

```
make
```

## Running

### Growing A Model

```
etgrow
```

### Applying A Model

```
etpredict
```
