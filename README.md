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

## Usage

1. You provide training data.
  * Training data is a list of sets of values.
  * One set of values is one or more independent variables, and exactly one dependent variable.
2. You format the data into an ASCII (text) file following the comma-separated-value format (CSV).
  * The first line of the file is an ordered list of variable **names**.
  * Each following line represents one **example**, which is a complete set of **values** in the same order as the variable names.
3. You run `etgrow` using that file as a **training** file, and specifying where you'd like the output **model** to go.
4. If you have more data for which the **dependent variable** is unknown, then you run `etpredict` using the **model** from `etgrow` to make predictions about the values of the dependent variable.

## Growing A Model

```
etgrow
```

## Applying A Model

```
etpredict
```
