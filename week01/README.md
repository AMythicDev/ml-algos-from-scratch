## Iteration Equation for Gradient Descent
$$
x_{i+1} = x_{i} - \eta \nabla f(x_{i})
$$

## Iteration Equation for Accelerated Gradient Descent
Using $y$ as the extrapolation point:

$$
x_{t+1} = y_{t} - \eta \nabla f(y_{t}) \\
$$

$$
y_{t+1} = x_{t+1} + \beta (x_{t+1} - x_{t})
$$

where $\beta = \frac{\sqrt{ \beta } - \sqrt{ \alpha }}{\sqrt{ \beta } + \sqrt{ \alpha }}$ for in case of strongly convex functions.

Using $v_{t}$ velocity vector:

$$
v_{t+1} = \beta v_{t} - \eta \nabla f(x_{t} + \beta v_{t}) \\
$$

$$
x_{t+1} = x_{t} + v_{t+1}
$$

## Iteration Equation for Momentum method

$$
v_{t+1} = \mu v_{t} - \eta \nabla f(x_{t}) \\
$$

$$
x_{t+1} = x_{t} + v_{t+1}
$$

## Value of $\gamma$ in GD and AGD
- GD: $\gamma = \frac{\alpha}{\alpha + \beta}$
- AGD: $\gamma = \frac{1}{\sqrt{ \frac{\beta}{\alpha} - 1 }}$
