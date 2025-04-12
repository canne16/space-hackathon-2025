use std::io::Write;

mod solvers;

use crate::solvers::*;

const mu: f64 = 398600.4415e9;
const R: f64 = 6371302.0;
const omega_E: f64 = 7.29211e-5;
const J_2: f64 = 1082.8e-6;

fn write_csv<const N: usize>(solution: &CauchySolution<N>, path: String) {
    let mut file = std::fs::File::create(format!("{}.csv", path)).expect("Failed to open file");
    file.write("Time, X, Y, Z, VX, VY, VZ\n".as_bytes())
        .expect("failed to write to file");
    for i in 0..solution.t.len() {
        file.write(format!("{:>10.3}", solution.t[i]).as_bytes())
            .expect("failed to write to file");
        for j in 0..N {
            file.write(format!(", {:>18.6}", solution.x[i][j] / 1000.0).as_bytes())
                .expect("failed to write to file");
        }
        file.write("\n".as_bytes())
            .expect("failed to write to file");
    }
}

fn find_last_start_intersection(solution: &CauchySolution<6>) -> usize {
    let mut res: usize = 0;
    let mut closest_longitude: f64 = 100.0;
    for i in 1..(solution.t.len() - 1) {
        if solution.x[i][2].abs() > solution.x[i - 1][2].abs() {
            continue;
        }
        if solution.x[i][2].abs() > solution.x[i + 1][2].abs() {
            continue;
        }
        if solution.x[i][2] < solution.x[i - 1][2] {
            continue;
        }
        if solution.x[i][2] > solution.x[i + 1][2] {
            continue;
        }
        if (closest_longitude >= solution.x[i][1].atan2(solution.x[i][0]).abs())
        {
            res = i;
            closest_longitude = solution.x[i][1].atan2(solution.x[i][0]).abs();
        }
    }
    res
}

// a = [x, y, z, v_x, v_y, v_z]
fn gravity_law(_: f64, a: &[f64; 6]) -> [f64; 6] {
    let r = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [
        a[3],
        a[4],
        a[5],
        -a[0] * mu / r.powf(3.0),
        -a[1] * mu / r.powf(3.0),
        -a[2] * mu / r.powf(3.0),
    ]
}

pub struct CauchyProblem<const N: usize> {
    pub f: fn(t: f64, &[f64; N]) -> [f64; N],
    pub start: f64,
    pub stop: f64,
    pub x_0: [f64; N],
}

fn calc_velocity(r: f64) -> f64 {
    (mu / r).sqrt()
}

fn main() {
    // let r: f64 = R + 50000.0;
    const orbit_angle: f64 = 98.0 * std::f64::consts::PI / 180.0;

    let mut solver: solvers::RungeKuttaMethod<6, 3, 18> = solvers::RungeKuttaMethod::new(
        4,
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-1.0, 2.0, 0.0]],
        [1f64 / 6f64, 2f64 / 3f64, 1f64 / 6f64],
        [0f64, 0.5f64, 1f64],
        "Kutta's third-order method (Explicit)".to_string(),
    );

    fn equation(x: &[f64; 2]) -> [f64; 2] {
        let r: f64 = x[0];
        // let v: f64 = calc_velocity(r);
        let v: f64 = x[1];
        // let t: f64 = x[1];
        let problem: solvers::CauchyProblem<6> = solvers::CauchyProblem {
            f: gravity_law,
            start: 0.0,

            stop: 100.0 * 60.0 * 60.0,
            // stop: t.min(100.0 * 60.0 * 60.0),
            x_0: [
                r,
                0.0,
                0.0,
                0.0,
                v * orbit_angle.cos(),
                v * orbit_angle.sin(),
            ],
        };

        let mut solver: solvers::RungeKuttaMethod<6, 3, 18> = solvers::RungeKuttaMethod::new(
            4,
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-1.0, 2.0, 0.0]],
            [1f64 / 6f64, 2f64 / 3f64, 1f64 / 6f64],
            [0f64, 0.5f64, 1f64],
            "Kutta's third-order method (Explicit)".to_string(),
        );

        let (mut solution, res) = solver.solve(&problem, 0.1, false, Some(1));

        for i in 0..solution.t.len() {
            let t = solution.t[i];
            let angle = -t * omega_E;
            solution.x[i] = [
                solution.x[i][0] * angle.cos() - solution.x[i][1] * angle.sin(),
                solution.x[i][0] * angle.sin() + solution.x[i][1] * angle.cos(),
                solution.x[i][2],
                0.0,
                0.0,
                0.0,
            ];
        }

        println!("Last intersection: {}", find_last_start_intersection(&solution));
        let x = solution.x[find_last_start_intersection(&solution)];

        [x[1].atan2(x[0]) * R, (x[2])]
    }

    let r_0 = R + 500000.0;

    let [r, v] = solvers::solve_newton(equation, &[r_0, calc_velocity(r_0)], None, true).unwrap();
    // let v = calc_velocity(r);
    let t: f64 = 100.0 * 60.0 * 60.0;

    println!("Optimal orbit: {:?}", r - R);

    // let velocity: f64 = calc_velocity(r);
    let problem: solvers::CauchyProblem<6> = solvers::CauchyProblem {
        f: gravity_law,
        start: 0.0,

        stop: t,
        x_0: [
            r,
            0.0,
            0.0,
            0.0,
            v * orbit_angle.cos(),
            v * orbit_angle.sin(),
        ],
    };

    let (solution, res) = solver.solve(&problem, 0.1, true, Some(600));

    let mut earth_solution: solvers::CauchySolution<6> = solvers::CauchySolution {
        t: vec![],
        x: vec![],
        method_name: solution.method_name.clone(),
    };

    for i in 0..solution.t.len() {
        let t = solution.t[i];
        let angle = -t * omega_E;
        earth_solution.t.push(t);
        earth_solution.x.push([
            solution.x[i][0] * angle.cos() - solution.x[i][1] * angle.sin(),
            solution.x[i][0] * angle.sin() + solution.x[i][1] * angle.cos(),
            solution.x[i][2],
            0.0,
            0.0,
            0.0,
        ]);
    }

    write_csv(
        &earth_solution,
        "../../problem_data/plot_data/static_earth_orbit".to_string(),
    );
    write_csv(
        &solution,
        "../../problem_data/plot_data/static_space_orbit".to_string(),
    );

    println!("equation(solution): {:?}", equation(&[r, t]));
    println!("solution[-1]: {:?}", *earth_solution.x.last().unwrap());
    println!(
        "Last intersection: {}",
        find_last_start_intersection(&earth_solution)
    );
}
