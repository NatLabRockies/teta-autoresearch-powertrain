# Domain

## Context

Our training data represents simulated vehicle runs over drive cycle traces (typically 1hz).
At each point we simulate the vehicle dynamics and get an energy estimation for that point.
We take these point level results and aggregate them up to the trip/road segment level.
Then, the road segments have attributes like total distance, average speed, average road gradiant, time to traverse, etc.

This tree targets a **BEV (battery electric)** vehicle, the 2017 Chevy Bolt. Link energy can be
**negative** (regenerative braking), so the target distribution is asymmetric and heavy-tailed.

## Model Inference Environment

Note that when we're applying these models for inference, we only have limited data (which is why we're developing these models in the first place).
Our inference environment has the following features:

- Average Speed: The speed is given as an average over the link, either from assuming the speed driven is the posted speed, or, using probe data to gather average speeds.
- Average Road Gradient: The gradient is an average over the link, either from taking the elevation difference of the link endpoints, or, using probe data to sample the gradient at subpoints on a link.
- Link Distance: The distance of the link
- Geometry: The link geometry in the well known binary format using the 4326 CRS (latitude and longitude points)

Think about the inference environment as applying these models during a shortest path search in Google Maps where we only have limited information.
If you're considering any kind of link sequencing, we will only have the context of the previous links that have been traversed and know nothing about the future links that might be traversed.

Acceleration and driver behavior is known to have significant impact on how much energy a vehicle uses.
Unfortunately, we can't explictly capture acceleration and driver behavior in our current inference environment since we only have link level measuresments.

If you're considering any kind of link sequencing, we only want to include a single one link lookback.
While it wouldn't vilolate any correctness for including the next link in a sequence during our shortest path searches in RouteE Compass, it would cost significant engieering time and so we just want to optimzie models that don't consider the future link.
While it might be beneficial to include more links in the look back, our current search harness would incur a huge memory penalty for trying to enumerate new labels with that much previous context.
To that end, we want to limit the lookback to a single link and try to squeeze out as much accuracy as we can.

In addition, in RouteE Compass, we have competing objectives of performance and arruracy.
We want to come up with a model that gives us as much accuracy as we can without being very expensive to apply inference.
We run the energy inference at every link traversal in the shortest path search and so we need to be performant.

## What counts as better

`harness.evaluate()` returns a dict of trip and link RMSE metrics. A change is a keep only if it Pareto-dominates the current best on both of those metrics.

## Constraints

Do not filter or remove data points to reduce error. The model must be able to predict all values in the dataset, including extreme energy rates such as heavy regenerative braking. Filtering outliers artificially lowers RMSE without improving the model's actual predictive capability — we need accurate predictions across the full distribution.

Do not include a feature like link position since at inference time, we will not know the position of a link relative to a whole trajectory.
