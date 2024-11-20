myidx = h5read("omp-results.h5", '/idx');
myidx = myidx';
mydist = h5read("omp-results.h5", '/dist');
mydist = mydist';

train = h5read('mnist-784-euclidean.hdf5', '/train');
train = train';
test = h5read('mnist-784-euclidean.hdf5', '/test');
test = test';

tic;
[idx, dist] = knnsearch(train, test, 'K', 100);
toc;

% Number of queries and neighbors
[n_queries, k] = size(idx);

% Initialize overlap count vector
overlap_counts = zeros(n_queries, 1);

for i = 1:n_queries
    overlap_counts(i) = numel(intersect(idx(i, :), myidx(i, :)));
end

% Calculate average overlap
average_overlap = mean(overlap_counts);

% Display Average Number of Overlapping Neighbors per Query Results
fprintf('Average number of overlapping neighbors per query: %.2f out of %d\n', average_overlap, k);

% Compute relative difference
relative_diff = abs(mydist - dist) ./ dist;

% Calculate average relative difference per query
avg_relative_diff_per_query = mean(relative_diff, 2);

% Overall average relative difference
overall_avg_relative_diff = mean(avg_relative_diff_per_query);

% Display Overall Average Relative Difference
fprintf('Overall average relative difference in distances: %.2f%%\n', overall_avg_relative_diff * 100);

% Define a tolerance level (e.g., 5%)
tolerance = 0.05;

% Determine distances within tolerance
within_tolerance = relative_diff <= tolerance;

% Calculate percentage of distances within tolerance
percent_within_tolerance = (sum(within_tolerance, 'all') / numel(within_tolerance)) * 100;

% Display Percentage Within Tolerance Results
fprintf('Percentage of distances within %.2f%% tolerance: %.2f%%\n', tolerance*100, percent_within_tolerance);

% Initialize precision vector
precision_at_k = zeros(n_queries, 1);

for i = 1:n_queries
    % Intersection of neighbor indices
    common_neighbors = intersect(idx(i, :), myidx(i, :));
    precision_at_k(i) = numel(common_neighbors) / k;
end
average_precision_at_k = mean(precision_at_k);

% Display Average Precision at k Results
fprintf('Average precision at k: %.2f%%\n', average_precision_at_k * 100);