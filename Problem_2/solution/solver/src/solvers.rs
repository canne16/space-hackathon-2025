#![allow(dead_code)]
use std::{f64, vec};

fn unflatten<const A: usize, const B: usize>(flat: &[f64]) -> [[f64; A]; B] {
    assert!(flat.len() == A * B);
    std::array::from_fn(|i| std::array::from_fn(|j| flat[i * A + j]))
}

fn close_enough(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

fn close_enough_arr<const N: usize>(a: &[f64; N], b: &[f64; N], epsilon: f64) -> bool {
    let mut sum: f64 = 0f64;
    for i in 0..N {
        sum += (a[i] - b[i]).abs();
    }

    close_enough(sum, 0f64, epsilon)
}

pub fn derivative<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + h) - f(x - h)) / (2f64 * h)
}

pub fn partial_derivative<F, const N: usize>(f: F, x: &[f64; N], i: usize, j: usize, h: f64) -> f64
where
    F: Fn(&[f64; N]) -> [f64; N],
{
    let mut x_1: [f64; N] = x.clone();
    let mut x_2: [f64; N] = x.clone();

    x_1[j] += h;
    x_2[j] -= h;

    (f(&x_1)[i] - f(&x_2)[i]) / (2f64 * h)
}

pub fn solve_linear_system<const N: usize>(a: &[[f64; N]; N], b: &[f64; N]) -> [f64; N] {
    let mut l: [[f64; N]; N] = [[0f64; N]; N];
    let mut u: [[f64; N]; N] = [[0f64; N]; N];

    for i in 0..N {
        l[i][i] = 1f64;
        for j in i..N {
            let mut sum: f64 = 0f64;
            for k in 0..i {
                sum += l[i][k] * u[k][j];
            }
            u[i][j] = a[i][j] - sum;
        }

        for j in i..N {
            let mut sum: f64 = 0f64;
            for k in 0..i {
                sum += l[j][k] * u[k][i];
            }
            l[j][i] = (a[j][i] - sum) / u[i][i];
        }
    }

    let mut v: [f64; N] = [0f64; N];
    for i in 0..N {
        v[i] = b[i];
        for j in 0..i {
            v[i] -= l[i][j] * v[j];
        }
    }

    let mut x: [f64; N] = [0f64; N];

    for i in (0..N).rev() {
        x[i] = v[i] / u[i][i];
        for j in i + 1..N {
            x[i] -= u[i][j] * x[j] / u[i][i];
        }
    }

    x
}

pub fn solve_newton<F, const N: usize>(
    f: F,
    x: &[f64; N],
    max_iterations: Option<u32>,
    print_progress: bool,
) -> Result<[f64; N], &'static str>
where
    F: Fn(&[f64; N]) -> [f64; N],
{
    let max_iterations: u32 = max_iterations.unwrap_or(100u32);

    let mut x: [f64; N] = x.clone();
    let mut iterations: u32 = 0u32;

    while !close_enough_arr(&f(&x), &[0f64; N], 1e3) {
        if iterations == max_iterations {
            return Err("Cannot solve system, too many iterations");
        }

        let mut jacobian: [[f64; N]; N] = [[0f64; N]; N];

        for i in 0..N {
            for j in 0..N {
                jacobian[i][j] = -partial_derivative(&f, &x, i, j, x[j] * 1e-3);
            }
        }

        let delta_x: [f64; N] = solve_linear_system(&jacobian, &f(&x));

        for i in 0..N {
            x[i] += delta_x[i] * 10e-1;
        }

        if print_progress {
            println!("x: {:?}, f(x): {:?}", x, f(&x));
        }

        iterations += 1;
    }

    Ok(x)
}

pub struct CauchyProblem<const N: usize> {
    pub f: fn(t: f64, &[f64; N]) -> [f64; N],
    pub start: f64,
    pub stop: f64,
    pub x_0: [f64; N],
}

pub struct CauchySolution<const N: usize> {
    pub t: std::vec::Vec<f64>,
    pub x: std::vec::Vec<[f64; N]>,
    pub method_name: String,
}

pub trait DifferentialEquationNumericMethod<const N: usize> {
    fn solve(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        print_progress: bool,
        save_every: Option<u32>,
    ) -> (CauchySolution<N>, Result<(), &'static str>);
    fn get_name(&self) -> String;
}

pub enum SolverType {
    Explicit,
    Implicit,
}

impl std::fmt::Display for SolverType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverType::Explicit => write!(f, "Explicit"),
            SolverType::Implicit => write!(f, "Implicit"),
        }
    }
}

pub struct RungeKuttaMethod<const N: usize, const M: usize, const MN: usize> {
    order: u32,
    solver_type: SolverType,
    a: [[f64; M]; M],
    b: [f64; M],
    c: [f64; M],
    name: String,
}

impl<const N: usize, const M: usize, const NM: usize> RungeKuttaMethod<N, M, NM> {
    pub fn new(order: u32, a: [[f64; M]; M], b: [f64; M], c: [f64; M], name: String) -> Self {
        let mut solver_type = SolverType::Explicit;
        for i in 0..M {
            for j in i..M {
                if a[i][j] != 0f64 {
                    solver_type = SolverType::Implicit;
                }
            }
        }

        Self {
            order,
            solver_type,
            a,
            b,
            c,
            name,
        }
    }

    fn step_explicit(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        t: f64,
        x: [f64; N],
    ) -> Result<(f64, [f64; N]), &'static str> {
        let mut k: [[f64; N]; M] = [[0f64; N]; M];
        for i in 0..M {
            let arg_1: f64 = t + tau * self.c[i];
            let mut arg_2: [f64; N] = [0f64; N];

            for a in 0..N {
                for j in 0..i {
                    arg_2[a] += self.a[i][j] * k[j][a];
                }
                arg_2[a] *= tau;
                arg_2[a] += x[a];
            }

            k[i] = (problem.f)(arg_1, &arg_2);
        }

        let mut res: [f64; N] = [0f64; N];
        for j in 0..N {
            for i in 0..M {
                res[j] += self.b[i] * k[i][j];
            }
            res[j] *= tau;
            res[j] += x[j];
        }

        Ok((t + tau, res))
    }

    fn step_implicit(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        t: f64,
        x: [f64; N],
    ) -> Result<(f64, [f64; N]), &'static str> {
        let equation = |k_0: &[f64; NM]| {
            let k_0: [[f64; N]; M] = unflatten::<N, M>(k_0);
            let mut k_i: [[f64; N]; M] = [[0.0; N]; M];

            for i in 0..M {
                let arg_1: f64 = t + tau * self.c[i];
                let mut arg_2: [f64; N] = [0f64; N];

                for a in 0..N {
                    for j in 0..M {
                        arg_2[a] += self.a[i][j] * k_0[j][a];
                    }
                    arg_2[a] *= tau;
                    arg_2[a] += x[a];
                }
                for a in 0..N {
                    k_i[i][a] = (problem.f)(arg_1, &arg_2)[a] - k_0[i][a];
                }
            }

            let k_i: [f64; NM] = (*k_i.as_flattened()).try_into().unwrap();

            k_i
        };

        let k = match solve_newton(equation, &[0f64; NM], None, false) {
            Ok(x) => unflatten::<N, M>(&x),
            Err(err) => {
                println!("Failed to solve, {}", err);
                return Err("Failed to solve");
            }
        };

        let mut res: [f64; N] = [0f64; N];
        for j in 0..N {
            for i in 0..M {
                res[j] += self.b[i] * k[i][j];
            }
            res[j] *= tau;
            res[j] += x[j];
        }

        Ok((t + tau, res))
    }

    fn step(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        t: f64,
        x: [f64; N],
    ) -> Result<(f64, [f64; N]), &'static str> {
        match self.solver_type {
            SolverType::Explicit => self.step_explicit(problem, tau, t, x),
            SolverType::Implicit => self.step_implicit(problem, tau, t, x),
        }
    }
}

impl<const N: usize, const M: usize, const NM: usize> DifferentialEquationNumericMethod<N>
    for RungeKuttaMethod<N, M, NM>
{
    fn solve(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        print_progress: bool,
        save_every: Option<u32>,
    ) -> (CauchySolution<N>, Result<(), &'static str>) {
        let mut t_i: f64 = problem.start;
        let mut x_i: [f64; N] = problem.x_0.clone();
        let mut iterations: u32 = 0u32;
        let save_every = save_every.unwrap_or(1);

        let mut solution: CauchySolution<N> = CauchySolution {
            t: vec![],
            x: vec![],
            method_name: self.get_name(),
        };

        solution.t.push(problem.start);
        solution.x.push(problem.x_0.clone());
        if print_progress {
            print!("t: {:>20.10}, iterations: {}", problem.start, iterations)
        }
        iterations += 1;

        while t_i < problem.stop {
            match self.step(&problem, tau, t_i, x_i) {
                Ok((t, x)) => {
                    t_i = t;
                    x_i = x.clone();
                    if iterations % save_every == 0 {
                        solution.t.push(t);
                        solution.x.push(x);
                        if print_progress {
                            print!("\rt: {:>20.10}, iterations: {}", t, iterations)
                        }
                    }
                }
                Err(a) => {
                    println!("Failed to solve, reason: {}", a);
                    return (solution, Err("Failed to solve"));
                }
            };
            iterations += 1;
        }

        if print_progress {
            println!("");
        }

        (solution, Ok(()))
    }

    fn get_name(&self) -> String {
        return format!(
            "{}, order: {}, type: {}",
            self.name.clone(),
            self.order,
            self.solver_type
        );
    }
}

pub struct AdamsMethod<const N: usize> {
    order: usize,
    solver_type: SolverType,
}

impl<const N: usize> AdamsMethod<N> {
    pub fn new(order: usize, solver_type: SolverType) -> Self {
        Self { order, solver_type }
    }

    fn step_explicit(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        t: &std::vec::Vec<f64>,
        x: &std::vec::Vec<[f64; N]>,
        order: usize,
    ) -> Result<(f64, [f64; N]), &'static str> {
        if t.len() < order {
            self.step_explicit(problem, tau, t, x, t.len())
        } else {
            let n = t.len() - 1;
            match order {
                1 => {
                    let mut x_i: [f64; N] = [0.0; N];
                    for i in 0..N {
                        x_i[i] += 1.0 * (problem.f)(t[n - 0], &x[n - 0])[i];
                        x_i[i] /= 1.0;
                        x_i[i] *= tau;
                        x_i[i] += x[n][i];
                    }
                    Ok((t[n] + tau, x_i))
                }
                2 => {
                    let mut x_i: [f64; N] = [0.0; N];
                    for i in 0..N {
                        x_i[i] += 3.0 * (problem.f)(t[n - 0], &x[n - 0])[i];
                        x_i[i] -= 1.0 * (problem.f)(t[n - 1], &x[n - 1])[i];
                        x_i[i] /= 2.0;
                        x_i[i] *= tau;
                        x_i[i] += x[n][i];
                    }
                    Ok((t[n] + tau, x_i))
                }
                3 => {
                    let mut x_i: [f64; N] = [0.0; N];
                    for i in 0..N {
                        x_i[i] += 23.0 * (problem.f)(t[n - 0], &x[n - 0])[i];
                        x_i[i] -= 16.0 * (problem.f)(t[n - 1], &x[n - 1])[i];
                        x_i[i] += 05.0 * (problem.f)(t[n - 2], &x[n - 2])[i];
                        x_i[i] /= 12.0;
                        x_i[i] *= tau;
                        x_i[i] += x[n][i];
                    }
                    Ok((t[n] + tau, x_i))
                }
                4 => {
                    let mut x_i: [f64; N] = [0.0; N];
                    for i in 0..N {
                        x_i[i] += 55.0 * (problem.f)(t[n - 0], &x[n - 0])[i];
                        x_i[i] -= 59.0 * (problem.f)(t[n - 1], &x[n - 1])[i];
                        x_i[i] += 37.0 * (problem.f)(t[n - 2], &x[n - 2])[i];
                        x_i[i] -= 09.0 * (problem.f)(t[n - 3], &x[n - 3])[i];
                        x_i[i] /= 24.0;
                        x_i[i] *= tau;
                        x_i[i] += x[n][i];
                    }
                    Ok((t[n] + tau, x_i))
                }
                _ => Err("No method with such order"),
            }
        }
    }

    fn step_implicit(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        t: &std::vec::Vec<f64>,
        x: &std::vec::Vec<[f64; N]>,
        order: usize,
    ) -> Result<(f64, [f64; N]), &'static str> {
        if t.len() < order {
            self.step_implicit(problem, tau, t, x, t.len())
        } else {
            let n: usize = t.len() - 1;
            match order {
                1 => {
                    let equation = |x_next: &[f64; N]| {
                        let mut x_i: [f64; N] = [0.0; N];

                        for i in 0..N {
                            x_i[i] += 1.0 * (problem.f)(t[n - 0] + tau, x_next)[i];
                            x_i[i] /= 1.0;
                            x_i[i] *= tau;
                            x_i[i] += x[n][i];
                            x_i[i] -= x_next[i];
                        }

                        x_i
                    };

                    match solve_newton(equation, &[0.0; N], None, false) {
                        Ok(x) => Ok((t[n] + tau, x)),
                        Err(err) => {
                            println!("Failed to solve, {}", err);
                            Err("Failed to solve")
                        }
                    }
                }
                2 => {
                    let equation = |x_next: &[f64; N]| {
                        let mut x_i: [f64; N] = [0.0; N];

                        for i in 0..N {
                            x_i[i] += 1.0 * (problem.f)(t[n - 0] + tau, x_next)[i];
                            x_i[i] += 1.0 * (problem.f)(t[n - 0], &x[n - 0])[i];
                            x_i[i] /= 2.0;
                            x_i[i] *= tau;
                            x_i[i] += x[n][i];
                            x_i[i] -= x_next[i];
                        }

                        x_i
                    };

                    match solve_newton(equation, &x[n - 1], None, false) {
                        Ok(x) => Ok((t[n] + tau, x)),
                        Err(err) => {
                            println!("Failed to solve, {}", err);
                            Err("Failed to solve")
                        }
                    }
                }
                3 => {
                    let equation = |x_next: &[f64; N]| {
                        let mut x_i: [f64; N] = [0.0; N];

                        for i in 0..N {
                            x_i[i] += 5.0 * (problem.f)(t[n - 0] + tau, x_next)[i];
                            x_i[i] += 8.0 * (problem.f)(t[n - 0], &x[n - 0])[i];
                            x_i[i] -= 1.0 * (problem.f)(t[n - 1], &x[n - 1])[i];
                            x_i[i] /= 12.0;
                            x_i[i] *= tau;
                            x_i[i] += x[n][i];
                            x_i[i] -= x_next[i];
                        }

                        x_i
                    };

                    match solve_newton(equation, &x[n - 1], None, false) {
                        Ok(x) => Ok((t[n] + tau, x)),
                        Err(err) => {
                            println!("Failed to solve, {}", err);
                            Err("Failed to solve")
                        }
                    }
                }
                4 => {
                    let equation = |x_next: &[f64; N]| {
                        let mut x_i: [f64; N] = [0.0; N];

                        for i in 0..N {
                            x_i[i] += 09.0 * (problem.f)(t[n - 0] + tau, x_next)[i];
                            x_i[i] += 19.0 * (problem.f)(t[n - 0], &x[n - 0])[i];
                            x_i[i] -= 05.0 * (problem.f)(t[n - 1], &x[n - 1])[i];
                            x_i[i] += 01.0 * (problem.f)(t[n - 2], &x[n - 2])[i];
                            x_i[i] /= 24.0;
                            x_i[i] *= tau;
                            x_i[i] += x[n][i];
                            x_i[i] -= x_next[i];
                        }

                        x_i
                    };

                    match solve_newton(equation, &x[n - 1], None, false) {
                        Ok(x) => Ok((t[n] + tau, x)),
                        Err(err) => {
                            println!("Failed to solve, {}", err);
                            Err("Failed to solve")
                        }
                    }
                }
                _ => Err("No method with such order"),
            }
        }
    }
}

impl<const N: usize> DifferentialEquationNumericMethod<N> for AdamsMethod<N> {
    fn solve(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        print_progress: bool,
        save_every: Option<u32>,
    ) -> (CauchySolution<N>, Result<(), &'static str>) {
        let mut t_i: Vec<f64> = vec![problem.start];
        let mut x_i: Vec<[f64; N]> = vec![problem.x_0.clone()];
        let mut iterations: u32 = 0u32;
        let save_every = save_every.unwrap_or(1);

        let mut solution: CauchySolution<N> = CauchySolution {
            t: vec![],
            x: vec![],
            method_name: self.get_name(),
        };

        solution.t.push(problem.start);
        solution.x.push(problem.x_0.clone());
        if print_progress {
            print!("t: {:>20.10}, iterations: {}", problem.start, iterations)
        }
        iterations += 1;

        while *t_i.last().unwrap() < problem.stop {
            let res: Result<(f64, [f64; N]), &str> = match self.solver_type {
                SolverType::Explicit => self.step_explicit(&problem, tau, &t_i, &x_i, self.order),
                SolverType::Implicit => self.step_implicit(&problem, tau, &t_i, &x_i, self.order),
            };

            match res {
                Ok((t, x)) => {
                    t_i.push(t);
                    x_i.push(x);
                    if t_i.len() > self.order {
                        t_i.remove(0);
                        x_i.remove(0);
                    }
                    if iterations % save_every == 0 {
                        solution.t.push(t);
                        solution.x.push(x);
                        if print_progress {
                            print!("\rt: {:>20.10}, iterations: {}", t, iterations)
                        }
                    }
                }
                Err(a) => {
                    println!("Failed to solve, reason: {}", a);
                    return (solution, Err("Failed to solve"));
                }
            };
            iterations += 1;
        }

        if print_progress {
            println!("");
        }

        (solution, Ok(()))
    }

    fn get_name(&self) -> String {
        return format!("{} Adams method of order: {}", self.solver_type, self.order,);
    }
}

pub struct BackwardDifferentiationMethod<const N: usize> {
    order: usize,
    solver_type: SolverType,
}

impl<const N: usize> BackwardDifferentiationMethod<N> {
    pub fn new(order: usize, solver_type: SolverType) -> Self {
        Self { order, solver_type }
    }

    fn step_explicit(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        t: &std::vec::Vec<f64>,
        x: &std::vec::Vec<[f64; N]>,
        order: usize,
    ) -> Result<(f64, [f64; N]), &'static str> {
        if t.len() < order {
            self.step_explicit(problem, tau, t, x, t.len())
        } else {
            /*
            n = len(x) - 1
            match N:
                case 1:
                case 2:
                case 3:
                case 4:
                */
            let n = t.len() - 1;
            match order {
                1 => {
                    let mut x_i: [f64; N] = [0.0; N];
                    for i in 0..N {
                        x_i[i] += (problem.f)(t[n], &x[n])[i];
                        x_i[i] *= 1.0 * tau / 1.0;
                        x_i[i] += 1.0 * x[n][i] / 1.0;
                        x_i[i] *= 1.0 / 1.0;
                    }
                    Ok((t[n] + tau, x_i))
                }
                2 => {
                    let mut x_i: [f64; N] = [0.0; N];
                    for i in 0..N {
                        x_i[i] += (problem.f)(t[n], &x[n])[i];
                        x_i[i] *= 2.0 * tau / 1.0;
                        x_i[i] += 1.0 * x[n - 1][i] / 1.0;
                        x_i[i] *= 1.0 / 1.0;
                    }
                    Ok((t[n] + tau, x_i))
                }
                3 => {
                    let mut x_i: [f64; N] = [0.0; N];
                    for i in 0..N {
                        x_i[i] += (problem.f)(t[n], &x[n])[i];
                        x_i[i] *= 1.0 * tau / 1.0;
                        x_i[i] -= 1.0 * x[n - 0][i] / 2.0;
                        x_i[i] += 1.0 * x[n - 1][i] / 1.0;
                        x_i[i] -= 1.0 * x[n - 2][i] / 6.0;
                        x_i[i] *= 3.0 / 1.0;
                    }
                    Ok((t[n] + tau, x_i))
                }
                _ => Err("No method with such order"),
            }
        }
    }

    fn step_implicit(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        t: &std::vec::Vec<f64>,
        x: &std::vec::Vec<[f64; N]>,
        order: usize,
    ) -> Result<(f64, [f64; N]), &'static str> {
        if t.len() < order {
            self.step_implicit(problem, tau, t, x, t.len())
        } else {
            let n: usize = t.len() - 1;
            match order {
                1 => {
                    let equation = |x_next: &[f64; N]| {
                        let mut x_i: [f64; N] = [0.0; N];

                        for i in 0..N {
                            x_i[i] += (problem.f)(t[n] + tau, x_next)[i];
                            x_i[i] *= tau;
                            x_i[i] -= 1.0 * x_next[i] / 1.0;
                            x_i[i] += 1.0 * x[n - 0][i] / 1.0;
                        }

                        x_i
                    };

                    match solve_newton(equation, &[0.0; N], None, false) {
                        Ok(x) => Ok((t[n] + tau, x)),
                        Err(err) => {
                            println!("Failed to solve, {}", err);
                            Err("Failed to solve")
                        }
                    }
                }
                2 => {
                    let equation = |x_next: &[f64; N]| {
                        let mut x_i: [f64; N] = [0.0; N];

                        for i in 0..N {
                            x_i[i] += (problem.f)(t[n] + tau, x_next)[i];
                            x_i[i] *= tau;
                            x_i[i] -= 3.0 * x_next[i] / 2.0;
                            x_i[i] += 2.0 * x[n - 0][i] / 1.0;
                            x_i[i] -= 1.0 * x[n - 1][i] / 2.0;
                        }

                        x_i
                    };

                    match solve_newton(equation, &x[n - 1], None, false) {
                        Ok(x) => Ok((t[n] + tau, x)),
                        Err(err) => {
                            println!("Failed to solve, {}", err);
                            Err("Failed to solve")
                        }
                    }
                }
                3 => {
                    let equation = |x_next: &[f64; N]| {
                        let mut x_i: [f64; N] = [0.0; N];

                        for i in 0..N {
                            x_i[i] += (problem.f)(t[n] + tau, x_next)[i];
                            x_i[i] *= tau;
                            x_i[i] -= 11.0 * x_next[i] / 6.0;
                            x_i[i] += 3.0 * x[n - 0][i] / 1.0;
                            x_i[i] -= 3.0 * x[n - 1][i] / 2.0;
                            x_i[i] += 1.0 * x[n - 2][i] / 3.0;
                        }

                        x_i
                    };

                    match solve_newton(equation, &x[n - 1], None, false) {
                        Ok(x) => Ok((t[n] + tau, x)),
                        Err(err) => {
                            println!("Failed to solve, {}", err);
                            Err("Failed to solve")
                        }
                    }
                }
                4 => {
                    let equation = |x_next: &[f64; N]| {
                        let mut x_i: [f64; N] = [0.0; N];

                        for i in 0..N {
                            x_i[i] += (problem.f)(t[n] + tau, x_next)[i];
                            x_i[i] *= tau;
                            x_i[i] -= 25.0 * x_next[i] / 11.0;
                            x_i[i] += 4.0 * x[n - 0][i] / 1.0;
                            x_i[i] -= 3.0 * x[n - 1][i] / 1.0;
                            x_i[i] += 4.0 * x[n - 2][i] / 3.0;
                            x_i[i] -= 1.0 * x[n - 3][i] / 4.0;
                        }

                        x_i
                    };

                    match solve_newton(equation, &x[n - 1], None, false) {
                        Ok(x) => Ok((t[n] + tau, x)),
                        Err(err) => {
                            println!("Failed to solve, {}", err);
                            Err("Failed to solve")
                        }
                    }
                }
                _ => Err("No method with such order"),
            }
        }
    }
}

impl<const N: usize> DifferentialEquationNumericMethod<N> for BackwardDifferentiationMethod<N> {
    fn solve(
        &mut self,
        problem: &CauchyProblem<N>,
        tau: f64,
        print_progress: bool,
        save_every: Option<u32>,
    ) -> (CauchySolution<N>, Result<(), &'static str>) {
        let mut t_i: Vec<f64> = vec![problem.start];
        let mut x_i: Vec<[f64; N]> = vec![problem.x_0.clone()];
        let mut iterations: u32 = 0u32;
        let save_every = save_every.unwrap_or(1);

        let mut solution: CauchySolution<N> = CauchySolution {
            t: vec![],
            x: vec![],
            method_name: self.get_name(),
        };

        solution.t.push(problem.start);
        solution.x.push(problem.x_0.clone());
        if print_progress {
            print!("t: {:>20.10}, iterations: {}", problem.start, iterations)
        }
        iterations += 1;

        while *t_i.last().unwrap() < problem.stop {
            let res: Result<(f64, [f64; N]), &str> = match self.solver_type {
                SolverType::Explicit => self.step_explicit(&problem, tau, &t_i, &x_i, self.order),
                SolverType::Implicit => self.step_implicit(&problem, tau, &t_i, &x_i, self.order),
            };

            match res {
                Ok((t, x)) => {
                    t_i.push(t);
                    x_i.push(x);
                    if t_i.len() > self.order {
                        t_i.remove(0);
                        x_i.remove(0);
                    }
                    if iterations % save_every == 0 {
                        solution.t.push(t);
                        solution.x.push(x);
                        if print_progress {
                            print!("\rt: {:>20.10}, iterations: {}", t, iterations)
                        }
                    }
                }
                Err(a) => {
                    println!("Failed to solve, reason: {}", a);
                    return (solution, Err("Failed to solve"));
                }
            };
            iterations += 1;
        }

        if print_progress {
            println!("");
        }

        (solution, Ok(()))
    }

    fn get_name(&self) -> String {
        return format!(
            "{} Backward differentiation method of order: {}",
            self.solver_type, self.order,
        );
    }
}

/// Tests

#[cfg(test)]
mod tests {
    use crate::solvers::*;

    #[test]
    fn test_solve_linear_system() {
        const N: usize = 3;
        let a = [
            [-4f64, 9f64, -4f64],
            [-5f64, -5f64, 6f64],
            [2f64, 5f64, -8f64],
        ];

        let b = [-64f64, 104.6f64, -85.2f64];

        let x = crate::solvers::solve_linear_system(&a, &b);
        for i in 0..N {
            let mut sum: f64 = 0f64;
            for j in 0..N {
                sum += a[i][j] * x[j];
            }
            assert!(close_enough(sum, b[i], 1e-6));
        }
    }

    #[test]
    fn test_derivative() {
        fn f(x: f64) -> f64 {
            x.sin()
        }

        assert!(close_enough(derivative(f, 0f64, 1e-6), 1f64, 1e-6));
        assert!(close_enough(
            derivative(f, std::f64::consts::FRAC_PI_2, 1e-6),
            0f64,
            1e-6
        ));
    }

    #[test]
    fn test_partial_derivative() {
        fn f(x: &[f64; 2]) -> [f64; 2] {
            [x[0] * x[0] + x[1] * x[1], 2f64 * x[0] * x[1]]
        }

        assert!(close_enough(
            partial_derivative(f, &[0f64, 0f64], 0, 0, 1e-6),
            0f64,
            1e-6
        ));
        assert!(close_enough(
            partial_derivative(f, &[0f64, 0f64], 0, 1, 1e-6),
            0f64,
            1e-6
        ));
        assert!(close_enough(
            partial_derivative(f, &[0f64, 0f64], 1, 0, 1e-6),
            0f64,
            1e-6
        ));
        assert!(close_enough(
            partial_derivative(f, &[0f64, 0f64], 1, 1, 1e-6),
            0f64,
            1e-6
        ));

        assert!(close_enough(
            partial_derivative(f, &[1f64, 1f64], 0, 0, 1e-6),
            2f64,
            1e-6
        ));
        assert!(close_enough(
            partial_derivative(f, &[1f64, 1f64], 0, 1, 1e-6),
            2f64,
            1e-6
        ));
        assert!(close_enough(
            partial_derivative(f, &[1f64, 1f64], 1, 0, 1e-6),
            2f64,
            1e-6
        ));
        assert!(close_enough(
            partial_derivative(f, &[1f64, 1f64], 1, 1, 1e-6),
            2f64,
            1e-6
        ));
    }

    #[test]
    fn test_solve_newton() {
        fn f(x: &[f64; 2]) -> [f64; 2] {
            [
                (x[0] + 1f64).sin() - x[1] - 1.2f64,
                2f64 * x[0] + x[1].cos() - 2f64,
            ]
        }

        let solution = solve_newton(f, &[0f64, 0f64], None, true).unwrap();

        assert!(close_enough_arr(&f(&solution), &[0f64, 0f64], 1e-6));
    }
}
