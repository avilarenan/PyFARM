# PyFARM - A Python Package for Time Series Feature Alignment

**PyFARM** is a Python package that provides functions for comparing and aligning two time series. This package includes three core functions that allow you to measure the **distance** between time series, **mean product** for feature alignment, and the **slope-mean-factorial** quality measure for evaluating the relationship between two time series.

## Installation

You can install `PyFARM` using pip:

```bash
pip install PyFARM
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone https://github.com/avilarenan/PyFARM.git
cd PyFARM
pip install .
```

## Functions

### 1. `farm.dist()`

#### Description
The `farm.dist()` function calculates the **Farm distance** between two vectors (time series) based on their angular alignment. If both vectors have a positive or negative argument, the distance is the sine of the absolute angle between them. For contrasting vectors (e.g., one vector has a positive angle, and the other has a negative angle), the resulting distance is computed as `1 - exp(-phi * 5)`.

#### Parameters
- **dyr** (float): The reference time series' normalized delta (`dy`) relative to `dx = 1`.
- **dyq** (float, optional): The query time series' normalized delta (`dy`). This is only required if `dyr` does not already include both reference and query delta values.
- **metric.space** (bool, default=False): If `True`, the distance calculation relies on the sine function only, ensuring triangle inequality compliance.

#### Returns
- **float**: A distance value between the two time series.

#### Example
```python
from PyFARM import farm

# Example usage
dist = farm.farm_dist(0.23, -2.85)
print(dist)  # Output will be the computed distance
```

---

### 2. `farm.mean.p()`

#### Description
The `farm.mean.p()` function computes a quality measure for the alignment of two time series based on the **mean absolute product** of their corresponding values. The formula used is:

```math
\text{mean\_p} = \left|\frac{1}{n} \sum_{i=0}^{n} \text{ref}_i \cdot \text{qry}_i \right|
```

A higher value indicates better alignment between the two time series.

#### Parameters
- **rts** (list or np.array): The reference time series (of length `n`).
- **qts** (list or np.array): The query time series (of length `n`).

#### Returns
- **float**: The quality measure, where a higher value indicates better alignment.

#### Example
```python
from PyFARM import farm

# Example time series
rts = [1, 2, 1.5, 2.1, 5, 2.32, 1, 0.2]
qts = [0.2, 0.4, 0.4, 0.5, 1, 0.92, 0.5, 0.2]

# Calculate the mean product quality measure
quality = farm.farm_mean_p(rts, qts)
print(quality)  # Output will be the computed mean product value
```

---

### 3. `farm.slm.f()`

#### Description
The `farm.slm.f()` function calculates the **slope-mean-factorial** quality measure for feature alignment between two time series. This measure evaluates the alignment based on the slopes (differences) of subsequent data points in the series, giving higher weights to larger differences. It also incorporates a penalty for changes in the time series' lengths (original vs. adjusted).

The formula used is:

```math
\text{slm\_f} = \frac{1}{n} \sum_{i=2}^{n} \Delta \text{ref}_{[i-1, i]} \cdot \Delta \text{qry}_{[i-1, i]} \div \left(1 + \left|\frac{\#smp_{\text{orig}} - \#smp_{\text{method}}}{\#smp_{\text{orig}}}\right|\right)
```

#### Parameters
- **rts** (list or np.array): The reference time series (of length `n`).
- **qts** (list or np.array): The query time series (of length `n`).
- **l.orig** (int, default=1): The length of the original time series.
- **l.adj** (int, default=1): The length of the adjusted time series.

#### Returns
- **float**: A quality measure indicating the alignment between the two time series.

#### Example
```python
from PyFARM import farm

# Example time series
rts = [1, 2, 1.5, 2.1, 5, 2.32, 1, 0.2]
qts = [0.2, 0.4, 0.4, 0.5, 1, 0.92, 0.5, 0.2]

# Calculate the slope-mean factorial quality measure
slm_quality = farm.farm_slm_f(rts, qts)
print(slm_quality)  # Output will be the computed quality measure
```

---

## Contributing

We welcome contributions! If you'd like to contribute to the development of `PyFARM`, please fork the repository, make your changes, and submit a pull request.

## License

`PyFARM` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
