use std::io::Write;

mod solvers;

use crate::solvers::*;

const mu: f64 = 398600.4415e9;
const R: f64 = 6371302.0;
const omega_E: f64 = 729211e-5;
const J_2: f64 = 1082.8e-6;

fn write_csv<const N: usize>(solution: &CauchySolution<N>, path: String) {
    let mut file =
        std::fs::File::create(format!("{}.csv", path)).expect("Failed to open file");
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
    let r: f64 = R + 500000.0;
    let angle: f64 = 98.0 * std::f64::consts::PI / 180.0;
    let velocity: f64 = calc_velocity(r);

    let problem: solvers::CauchyProblem<6> = solvers::CauchyProblem {
        f: gravity_law,
        start: 0.0,

        // Geostatic
        // stop: 60.0 * 60.0 * 24.0,
        // x_0: [42164000.0, 0.0, 0.0, 0.0, 3070.0, 0.0],

        stop: 100.0 * 60.0 * 60.0,
        x_0: [r, 0.0, 0.0, 0.0, velocity * angle.cos(), velocity * angle.sin()],
    };

    let mut solver: solvers::RungeKuttaMethod<6, 3, 18> = solvers::RungeKuttaMethod::new(
        4,
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-1.0, 2.0, 0.0]],
        [1f64 / 6f64, 2f64 / 3f64, 1f64 / 6f64],
        [0f64, 0.5f64, 1f64],
        "Kutta's third-order method (Explicit)".to_string(),
    );

    let (solution, res) = solver.solve(&problem, 0.1, true, Some(1000));
    println!("{:?}", res);
    // write_csv(solution);

    let mut earth_solution: solvers::CauchySolution<6> = solvers::CauchySolution {
        t: vec![],
        x: vec![],
        method_name: solution.method_name.clone(),
    };

    for i in 0..solution.t.len() {
        let t = solution.t[i];
        let angle = t * omega_E;
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
    write_csv(&earth_solution, "../../problem_data/plot_data/static_earth_orbit".to_string());
    write_csv(&solution, "../../problem_data/plot_data/static_space_orbit".to_string());
}
