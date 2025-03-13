
# Term Project - Travel Planner [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

<a href="https://marp.app"><img src="https://marp.app/favicon.png" align="right" width="120" height="120" /></a>

> A travel planner provides you itiniery suggestions.

Eve Huang

[marp]: https://marp.app

## Agenda

- Introduction
  - Problem Definition
  - Idea
- Design and Algorithm
  - Design
  - Algorithm
- Results
  - Travel Suggestions

---

## Introduction

### Problem Definition

- I like to travel, and I enjoy doing research on planning my trips. However, not everyone enjoy doing so. Most of my friends as long as they have trips with me, usually I am the one who plan all of itinerary. Some of my friends hope to plan their trips by themselves while they have hard time to summarize the trip itinerary by themselves as too much information to be considered to plan trips.

### Idea

- A trip planner to provide route/trip duration/cost suggestions is beneficial for those people who have hard time to arrange trips by themselves.

---

## Design and Algorithm

### Design
- Elements for a trip planning are:
  - Places to go
  - Transporation methods
  - Transportation period
  - Cost of transporation

With above information, we can design a simple trip itinerary. Usually the major constraints for a trip are cost and time. Thus, this trip planner is desgined to provide route/duration/cost suggestions, letting users decide which route to go.

### Algorithm

- The code consists with:
  - Network graph visualization
  - Travel simulation
  - Multiple city travel simulation
  - Optimal travel plan simulation

Discrete event simulation(Simpy) is used in calculating the optimistic routes suggestions.

### How does it Works

- Program will list all places to go and the connections according to the list.
- Users choose the start point and destination.
- Program will simulate possible routes with travel duration and cost of transportation.

---

## Result

### Simulations

- With proper destination input, program can provide thorough route suggestions including transportation methods, duration and cost.

### Problems in the Program

- If the data connection is not well arranged, program will not be able to run the simulation.


