# Loop closure on fisheye images: appearance proposes, rays verify

Visual odometry can be locally accurate and still drift globally. Loop closure asks whether
the current camera has returned to an old place, then supplies a long-range constraint to the
pose graph. The difficult part is not finding images that look alike. It is accepting real
revisits without turning repeated wallpaper, doors, or desks into destructive false edges.

The runnable companion is `examples/12_loop_closure_tumvi.py`. It uses TUM-VI room1 and prints
precision/recall measurements rather than a similarity screenshot:

```bash
python examples/12_loop_closure_tumvi.py
```

## 1. The retrieval cascade

The script implements this deterministic cascade:

```text
ORB descriptors
  → hierarchical binary vocabulary
  → sparse TF-IDF image vector
  → inverted-file top candidate
  → temporal exclusion
  → calibrated bearing-ray RANSAC
  → precision/recall operating point
```

Bag of visual words deliberately removes descriptor order. For image \(d\), every ORB
descriptor is assigned to a leaf word. Counts become a sparse vector. The vocabulary is a
hierarchical k-means tree: with branching \(k\) and depth \(L\), its capacity is \(k^L\);
quantization compares at most \(kL\) binary centers, using Hamming distance.

For binary descriptors a cluster center is the majority bit at each position. An arithmetic
mean of descriptor bytes is not a valid ORB center.

## 2. Weight words by documents, not descriptors

Let \(n_{i,d}\) be word \(i\)'s count in image \(d\), \(N\) the number of vocabulary-training
images, and \(N_i\) the number of those images that contain the word:

\[
\operatorname{tf}_{i,d} = \frac{n_{i,d}}{\sum_j n_{j,d}}, \qquad
\operatorname{idf}_i = \log \frac{N}{N_i}.
\]

IDF uses **document frequency**. Repeating one descriptor many times inside one image must not
make the word look globally common. After TF-IDF weighting, the example L1-normalizes each
vector.

DBoW's L1 score then has three equivalent forms:

\[
s(v,w)=1-\frac{1}{2}\lVert v-w\rVert_1
      =\frac{1}{2}\sum_i(|v_i|+|w_i|-|v_i-w_i|)
      =\sum_i\min(v_i,w_i).
\]

It ranges from zero for disjoint supports to one for identical vectors. The inverted file
stores a posting list for each word and accumulates only the final expression above. It is
therefore exact for this score; it is not an approximate nearest-neighbor shortcut.

## 3. Freeze the vocabulary before evaluation

The official ten-image teaching demo trains and tests on the same images. That demonstrates
the API, but it cannot measure generalization: test descriptors affect both cluster centers
and IDF.

This example uses disjoint image indices:

- vocabulary frames: every 43rd source frame;
- retrieval/evaluation frames: every 20th source frame, excluding vocabulary frames;
- vocabulary, preprocessing, ORB configuration, and deterministic seed are fixed before a
  query is evaluated.

For a benchmark result, train on whole disjoint sequences and hold out the evaluation
trajectory. The Chapter 10 ARM reference does exactly that with KITTI sequences
02/06/07/08 for training and sequence 00 for evaluation.

## 4. Exclude trivial neighbors

At 20 Hz the frame immediately before a query usually wins. It is visually similar but is not
the long-range information a loop edge is meant to provide. The example excludes the newest
15 seconds from retrieval. In a real system this should be a keyframe/time policy, not a magic
image-count constant.

Repeated detections of the same revisit should also be grouped. Adding ten highly correlated
edges does not create ten independent pieces of information.

## 5. Raw fisheye pixels are not pinhole pixels

Appearance only proposes a candidate. The verifier:

1. Hamming-matches ORB descriptors and applies a ratio test.
2. Unprojects both pixel sets through the loaded TUM-VI camera calibration.
3. Estimates relative pose from unit bearing vectors with angular RANSAC.
4. Requires both an inlier count and an inlier ratio.

The essential constraint still holds on calibrated bearing vectors. Applying a pinhole
essential-matrix threshold directly to raw fisheye pixels silently changes the geometry,
especially near the image boundary.

Even geometric verification is not the final authority. A production SLAM system also checks
the proposed transform against odometry, the pose graph, or a mutually consistent loop
cluster. Robust pose-graph kernels are a final defence, not permission to accept weak loop
edges.

## 6. Choose the threshold from precision/recall

For a threshold sweep:

\[
\mathrm{precision}=\frac{TP}{TP+FP}, \qquad
\mathrm{recall}=\frac{TP}{TP+FN}.
\]

A missed loop leaves drift; a false loop can deform the whole map. Report the PR curve and
recall at zero observed false positives, then choose an operating point from the actual risk
budget. Do not report only the best-looking threshold on the evaluation sequence.

The example labels a pair from mocap position and orientation, reports zero-false-positive
recall and best F1 before and after ray verification, and uses stable tie-breaking throughout.

## 7. What belongs in DS-MSP

The example intentionally keeps the vocabulary implementation in the teaching layer:

- DS-MSP already owns the important wide-FOV seam—pixel-to-bearing calibration and robust
  relative pose on rays.
- DBoW3 is a mature production C++ retrieval engine; duplicating it in the core Python camera
  library would add maintenance and dependency cost.
- A future `ds_msp.vo` loop-closure service should accept a retrieval backend interface,
  enforce the temporal/geometric/graph-consistency policy, and remain camera-model agnostic.

That separation lets a deployment choose DBoW3, a learned global descriptor, or another
retriever while reusing the same calibrated verification and pose-graph contract.
