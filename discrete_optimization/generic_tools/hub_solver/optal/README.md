# Optal for discrete-optimization

We briefly explain how to make use of Optal wrappers coded in d-o.

## How to install Optal?

- npm + node >= 22.7.9: you need npm installed and node >= 22.7.9 as shown in package.json.
  See https://docs.npmjs.com/downloading-and-installing-node-js-and-npm for installation process.
- optal version: by default, package.json is set with the preview (free) version of Optal that returns
  lower bound and objective values but not the solution. If you have access to a full version, please modify
  accordingly package.json. For instance by replacing
  ```json
    "@scheduleopt/optalcp-bin-preview": "github:ScheduleOpt/optalcp-js-bin-preview#latest",
  ```
  with
  ```json
    "@scheduleopt/optalcp-bin-academic": "github:scheduleopt/optalcp-js-bin-academic#v2025.8.0",
  ```
- then go to this directory ("path/to/discrete-optimization/discrete_optimization/generic_tools/hub_solver/optal"), and type
    ```shell
    npm install
    ```

**Note**:
    If using the preview version of optal, make sure to use the option `do_not_retrieve_solutions=True` in `optal_solver.solve()`,
    where `optal_solver` is one of the optal wrapper included in d-o.
