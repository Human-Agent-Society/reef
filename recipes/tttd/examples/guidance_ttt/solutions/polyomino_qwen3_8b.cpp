#include <bits/stdc++.h>
using namespace std;
using Clock = chrono::steady_clock;

static Clock::time_point START_TIME = Clock::now();
static double elapsed() {
    return chrono::duration<double>(Clock::now() - START_TIME).count();
}

// Convex hull area using monotone chain (Andrew's algorithm)
static double convexHullArea(const vector<pair<int,int>>& cells) {
    if ((int)cells.size() < 3) return (double)cells.size();
    vector<pair<int,int>> pts = cells;
    sort(pts.begin(), pts.end());
    pts.erase(unique(pts.begin(), pts.end()), pts.end());
    int n = pts.size();
    if (n < 3) return (double)n;

    vector<pair<int,int>> hull(2 * n);
    int k = 0;
    for (int i = 0; i < n; i++) {
        while (k >= 2) {
            auto &a = hull[k - 2], &b = hull[k - 1];
            auto &c = pts[i];
            long long cross = (long long)(b.first - a.first) * (c.second - a.second)
                            - (long long)(b.second - a.second) * (c.first - a.first);
            if (cross <= 0) k--; else break;
        }
        hull[k++] = pts[i];
    }
    int lower = k + 1;
    for (int i = n - 2; i >= 0; i--) {
        while (k >= lower) {
            auto &a = hull[k - 2], &b = hull[k - 1];
            auto &c = pts[i];
            long long cross = (long long)(b.first - a.first) * (c.second - a.second)
                            - (long long)(b.second - a.second) * (c.first - a.first);
            if (cross <= 0) k--; else break;
        }
        hull[k++] = pts[i];
    }
    hull.resize(k - 1);

    double area = 0;
    int h = hull.size();
    for (int i = 0; i < h; i++) {
        int j = (i + 1) % h;
        area += (double)hull[i].first * hull[j].second
              - (double)hull[j].first * hull[i].second;
    }
    return fabs(area) / 2.0;
}

// Triple Exponential Moving Average for density tracking
struct TripleEMA {
    double ema1 = 0, ema2 = 0, ema3 = 0;
    double alpha = 0.35;
    bool initialized = false;

    void update(double value) {
        value = max(0.0, min(1.0, value));
        if (!initialized) {
            ema1 = ema2 = ema3 = value;
            initialized = true;
        } else {
            ema1 = alpha * value + (1.0 - alpha) * ema1;
            ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
            ema3 = alpha * ema2 + (1.0 - alpha) * ema3;
        }
    }

    double value() const {
        if (!initialized) return 0.5;
        double v = 3.0 * ema1 - 3.0 * ema2 + ema3;
        return max(0.0, min(1.0, v));
    }
};

struct Orient {
    vector<pair<int,int>> cells;
    int w, h;
    int minXf, minYf;
    int R, F;
    vector<pair<int,int>> colInfo; // (cx, maxCy) per unique column
};

struct Piece {
    vector<pair<int,int>> cells;
    vector<Orient> orients;
    int minOW = INT_MAX;
    int maxBA = 0;
    int minBA = INT_MAX;
    double irregularity = 0;
    double maxAspect = 0;
    double hullArea = 0;
    double hullRatio = 0;
    double shapeScore = 0;
    double fittingPotential = 0;
    double conflictScore = 0;
    double protrusionRatio = 0;
    double angularDiversity = 0;
    double rotationalSymmetry = 0;
    double gridAlignment = 0;
    double placementFlexibility = 0;
    double spaceFlexibility = 0;
    double avgAspect = 1.0;
    double minAspectOverOrients = 1e18;
    int minBW = INT_MAX, minBH = INT_MAX;
    int bestBW = 0, bestBH = 0;
    double minElongation = 1.0;
    double optimalAspect = 1.0;
    double compactAspect = 1.0;
};

// Dual-gap metric analysis result: depth metrics + shape compatibility + aspect-aware scores
// + placement pressure index for spatial-aware gap prioritization
// + predictive score for forward-looking gap impact estimation
// + multiStepPredictiveScore for multi-step residual forecasting
struct GapAnalysis {
    int activeCount;
    int passiveCount;
    double activeRatio;
    double totalGapArea;
    double shapeCompatScore;       // shape compatibility: fraction of piece dims fitting gaps
    double depthScore;              // normalized depth-based gap quality
    double dualScore;               // combined dual-gap metric
    double aspectCompatScore;      // aspect-ratio-weighted compatibility score
    double orientationAlignScore;  // orientation-specific alignment with piece optimal aspects
    double placementPressureScore; // spatial-aware placement pressure index
    double predictiveScore;        // forward-looking predictive gap impact score
    double multiStepPredictiveScore; // multi-step residual forecasting score
};

static vector<Orient> genOrients(const vector<pair<int,int>>& cells) {
    set<vector<pair<int,int>>> seen;
    vector<Orient> res;
    for (int F = 0; F < 2; ++F) {
        for (int R = 0; R < 4; ++R) {
            int minTx = INT_MAX, minTy = INT_MAX;
            int maxTx = INT_MIN, maxTy = INT_MIN;
            vector<pair<int,int>> tf;
            tf.reserve(cells.size());
            for (auto [x, y] : cells) {
                int sx = F ? -x : x;
                int sy = y;
                int rx = sx, ry = sy;
                for (int r = 0; r < R; ++r) {
                    int nrx = ry;
                    int nry = -rx;
                    rx = nrx;
                    ry = nry;
                }
                tf.push_back({rx, ry});
                minTx = min(minTx, rx); maxTx = max(maxTx, rx);
                minTy = min(minTy, ry); maxTy = max(maxTy, ry);
            }
            vector<pair<int,int>> norm;
            norm.reserve(tf.size());
            for (auto [x, y] : tf) norm.push_back({x - minTx, y - minTy});
            sort(norm.begin(), norm.end());
            if (seen.insert(norm).second) {
                map<int,int> colMax;
                for (auto [x, y] : norm) {
                    auto it = colMax.find(x);
                    if (it == colMax.end() || y > it->second) colMax[x] = y;
                }
                vector<pair<int,int>> colInfo(colMax.begin(), colMax.end());
                res.push_back(Orient{norm, maxTx - minTx + 1, maxTy - minTy + 1,
                                     minTx, minTy, R, F, colInfo});
            }
        }
    }
    return res;
}

static tuple<double,double,double> multiAspectEstimate(const vector<Piece>& pieces,
                                                long long totalCells) {
    double sumAspect = 0;
    double sumW = 0, sumH = 0;
    int count = 0;
    double weightedAspect = 0;
    double totalWeight = 0;
    double sumSqAspect = 0;

    double irrWeightedAspect = 0;
    double irrTotalWeight = 0;

    for (const auto& p : pieces) {
        double minArea = 1e18;
        double bestW = 0, bestH = 0;
        double bestAspect = 1.0;
        double bestSqDist = 1e18;

        double irrBestAspect = 1.0;
        double irrBestScore = -1e18;

        for (const auto& o : p.orients) {
            double area = (double)o.w * o.h;
            double aspect = (double)o.w / max(1, o.h);
            if (area < minArea || (area == minArea && aspect < bestAspect)) {
                minArea = area;
                bestW = o.w;
                bestH = o.h;
                bestAspect = aspect;
            }
            double sqDist = fabs(aspect - 1.0);
            if (sqDist < bestSqDist) {
                bestSqDist = sqDist;
            }
            double irrScore = sqDist * (1.0 + p.irregularity);
            if (irrScore > irrBestScore) {
                irrBestScore = irrScore;
                irrBestAspect = aspect;
            }
        }
        double weight = (double)p.cells.size();
        sumAspect += bestAspect;
        sumW += bestW;
        sumH += bestH;
        weightedAspect += bestAspect * weight;
        totalWeight += weight;
        sumSqAspect += (bestSqDist < 1e18) ? (1.0 - bestSqDist) : 1.0;
        count++;

        double irrW = p.irregularity + 0.1;
        irrWeightedAspect += irrBestAspect * irrW;
        irrTotalWeight += irrW;
    }

    double avgAspect = (count > 0) ? sumAspect / count : 1.0;
    double wAvgAspect = (totalWeight > 0) ? weightedAspect / totalWeight : 1.0;
    double areaMinAspect = (avgAspect + wAvgAspect) / 2.0;
    areaMinAspect = max(0.5, min(2.0, areaMinAspect));

    double sqBalAspect = (count > 0) ? sumSqAspect / count : 1.0;
    sqBalAspect = max(0.5, min(2.0, sqBalAspect));

    double irrAspect = (irrTotalWeight > 0) ? irrWeightedAspect / irrTotalWeight : 1.0;
    irrAspect = max(0.5, min(2.0, irrAspect));

    return {areaMinAspect, sqBalAspect, irrAspect};
}

static pair<int,int> estimateOptimalBoard(const vector<Piece>& pieces,
                                          long long totalCells,
                                          double aspect) {
    double sqrtArea = sqrt((double)totalCells);
    int estW = max(1, (int)round(sqrtArea * sqrt(aspect)));
    int estH = max(1, (int)ceil((double)totalCells / estW));
    return {estW, estH};
}

// Compute column heights from placement
static vector<int> computeColHeights(const vector<Piece>& pieces,
                                     const vector<array<int,4>>& place, int W) {
    vector<int> colH(W, 0);
    int n = (int)pieces.size();
    for (int i = 0; i < n; ++i) {
        int X = place[i][0], Y = place[i][1], R = place[i][2], F = place[i][3];
        for (auto [x, y] : pieces[i].cells) {
            int sx = F ? -x : x;
            int sy = y;
            int rx = sx, ry = sy;
            for (int r = 0; r < R; ++r) {
                int nrx = ry;
                int nry = -rx;
                rx = nrx;
                ry = nry;
            }
            int ax = X + rx, ay = Y + ry;
            if (ax >= 0 && ax < W) {
                colH[ax] = max(colH[ax], ay + 1);
            }
        }
    }
    return colH;
}

// Helper: compute placement pressure index for a single gap
static inline double computeGapPressureIndex(int gapWidth, int gapDepth,
                                              int leftH, int rightH,
                                              int skylineMax,
                                              const vector<pair<int,int>>& allPieceDims) {
    int totalDims = max(1, (int)allPieceDims.size());
    double tightnessSum = 0.0;
    int fitCount = 0;

    for (int di = 0; di < (int)allPieceDims.size(); ++di) {
        auto [pw, ph] = allPieceDims[di];
        if (pw <= gapWidth && ph <= gapDepth) {
            double tightW = (double)pw / max(1, gapWidth);
            double tightH = (double)ph / max(1, gapDepth);
            tightnessSum += tightW * tightH;
            fitCount++;
        }
        if (ph <= gapWidth && pw <= gapDepth) {
            double tightW = (double)ph / max(1, gapWidth);
            double tightH = (double)pw / max(1, gapDepth);
            tightnessSum += tightW * tightH;
            fitCount++;
        }
    }

    double tightnessAvg = (fitCount > 0) ? tightnessSum / (double)fitCount : 0.0;

    double posInfluence = (double)gapDepth / max(1, skylineMax);

    int neighborMax = max(leftH, rightH);
    int neighborMin = min(leftH, rightH);
    double neighborBalance = (neighborMax > 0)
        ? 1.0 - (double)abs(leftH - rightH) / max(1, neighborMax)
        : 0.5;

    double widthPressure = 1.0;
    if (gapWidth > 0) {
        int minFitW = INT_MAX;
        for (auto [pw, ph] : allPieceDims) {
            if (pw <= gapWidth && ph <= gapDepth) minFitW = min(minFitW, pw);
            if (ph <= gapWidth && pw <= gapDepth) minFitW = min(minFitW, ph);
        }
        if (minFitW < INT_MAX && gapWidth > 0) {
            widthPressure = (double)minFitW / (double)gapWidth;
        }
    }

    double pressureIndex = tightnessAvg * 0.5
                         + posInfluence * 0.25
                         + neighborBalance * 0.15
                         + widthPressure * 0.10;

    return max(0.0, min(1.0, pressureIndex));
}

// Predictive gap analysis: forward-looking heuristic estimating how placing
// the best-fitting piece in a gap would affect remaining space utilization.
static double computePredictiveScore(int gapWidth, int gapDepth,
                                       int leftH, int rightH,
                                       int skylineMax,
                                       const vector<pair<int,int>>& allPieceDims) {
    if (gapWidth <= 0 || gapDepth <= 0) return 0.0;

    int totalDims = max(1, (int)allPieceDims.size());

    int minPieceDim = INT_MAX;
    for (auto [pw, ph] : allPieceDims) {
        minPieceDim = min({minPieceDim, pw, ph});
    }
    if (minPieceDim == INT_MAX) minPieceDim = 1;

    double bestFillRatio = 0.0;
    int bestPw = 0, bestPh = 0;

    for (auto [pw, ph] : allPieceDims) {
        if (pw <= gapWidth && ph <= gapDepth) {
            double fillRatio = (double)(pw * ph) / (double)max(1, gapWidth * gapDepth);
            if (fillRatio > bestFillRatio) {
                bestFillRatio = fillRatio;
                bestPw = pw;
                bestPh = ph;
            }
        }
        if (ph <= gapWidth && pw <= gapDepth) {
            double fillRatio = (double)(pw * ph) / (double)max(1, gapWidth * gapDepth);
            if (fillRatio > bestFillRatio) {
                bestFillRatio = fillRatio;
                bestPw = ph;
                bestPh = pw;
            }
        }
    }

    if (bestPw == 0) return 0.0;

    int rightResidW = gapWidth - bestPw;
    int topResidH = gapDepth - bestPh;

    int rightFitCount = 0;
    if (rightResidW > 0) {
        for (auto [pw, ph] : allPieceDims) {
            if (pw <= rightResidW && ph <= gapDepth) rightFitCount++;
            if (ph <= rightResidW && pw <= gapDepth) rightFitCount++;
        }
    }

    int topFitCount = 0;
    if (topResidH > 0) {
        for (auto [pw, ph] : allPieceDims) {
            if (pw <= bestPw && ph <= topResidH) topFitCount++;
            if (ph <= bestPw && pw <= topResidH) topFitCount++;
        }
    }

    int cornerFitCount = 0;
    if (rightResidW > 0 && topResidH > 0) {
        for (auto [pw, ph] : allPieceDims) {
            if (pw <= rightResidW && ph <= topResidH) cornerFitCount++;
            if (ph <= rightResidW && pw <= topResidH) cornerFitCount++;
        }
    }

    double rightUsability = (rightResidW > 0) ? (double)rightFitCount / (2.0 * totalDims) : 0.0;
    double topUsability = (topResidH > 0) ? (double)topFitCount / (2.0 * totalDims) : 0.0;
    double cornerUsability = (rightResidW > 0 && topResidH > 0)
        ? (double)cornerFitCount / (2.0 * totalDims) : 0.0;

    double narrowPenalty = 0.0;
    if (rightResidW > 0 && rightResidW < minPieceDim) {
        narrowPenalty += 0.35;
    }
    if (topResidH > 0 && topResidH < minPieceDim) {
        narrowPenalty += 0.35;
    }

    double residualScore = rightUsability * 0.35 + topUsability * 0.30 + cornerUsability * 0.35;
    double predictiveScore = (1.0 - narrowPenalty) * residualScore * 0.5 + bestFillRatio * 0.5;

    return max(0.0, min(1.0, predictiveScore));
}

// Multi-step residual forecasting: simulate placing multiple pieces sequentially
// in predicted optimal positions within a gap. Instead of only evaluating the
// immediate single-step residual, this function tracks residual spaces across
// up to 3 sequential placements, measuring how well the space can be filled
// and how usable the remaining fragments remain over multiple steps.
static double computeMultiStepPredictiveScore(int gapWidth, int gapDepth,
                                                int leftH, int rightH,
                                                int skylineMax,
                                                const vector<pair<int,int>>& allPieceDims) {
    if (gapWidth <= 0 || gapDepth <= 0) return 0.0;

    int totalDims = max(1, (int)allPieceDims.size());
    int minPieceDim = INT_MAX;
    for (auto [pw, ph] : allPieceDims) {
        minPieceDim = min({minPieceDim, pw, ph});
    }
    if (minPieceDim == INT_MAX) minPieceDim = 1;

    // Simulate up to 3 steps of placement
    // Each step: find best piece for the largest remaining residual space,
    // place at bottom-left, split into right and top residuals
    double totalFillScore = 0.0;
    double totalResidualUsability = 0.0;
    int steps = 0;
    int maxSteps = 3;

    // Current spaces to explore: list of (width, depth) rectangles
    vector<pair<int,int>> spaces;
    spaces.push_back({gapWidth, gapDepth});

    for (int step = 0; step < maxSteps && !spaces.empty(); ++step) {
        // Find the space with largest area
        int bestSpaceIdx = 0;
        int bestSpaceArea = 0;
        for (int si = 0; si < (int)spaces.size(); ++si) {
            int area = spaces[si].first * spaces[si].second;
            if (area > bestSpaceArea) {
                bestSpaceArea = area;
                bestSpaceIdx = si;
            }
        }

        int sw = spaces[bestSpaceIdx].first;
        int sh = spaces[bestSpaceIdx].second;
        if (sw <= 0 || sh <= 0) {
            spaces.erase(spaces.begin() + bestSpaceIdx);
            continue;
        }

        // Find best-fitting piece for this space (highest fill ratio)
        double bestFillRatio = 0.0;
        int bestPw = 0, bestPh = 0;
        for (auto [pw, ph] : allPieceDims) {
            if (pw <= sw && ph <= sh) {
                double fr = (double)(pw * ph) / (double)max(1, sw * sh);
                if (fr > bestFillRatio) {
                    bestFillRatio = fr;
                    bestPw = pw;
                    bestPh = ph;
                }
            }
            if (ph <= sw && pw <= sh) {
                double fr = (double)(pw * ph) / (double)max(1, sw * sh);
                if (fr > bestFillRatio) {
                    bestFillRatio = fr;
                    bestPw = ph;
                    bestPh = pw;
                }
            }
        }

        if (bestPw == 0) {
            // No piece fits this space - skip it
            spaces.erase(spaces.begin() + bestSpaceIdx);
            continue;
        }

        steps++;
        totalFillScore += bestFillRatio;

        // Compute residual spaces after placing bestPw x bestPh at bottom-left
        int rightResidW = sw - bestPw;
        int topResidH = sh - bestPh;

        // Remove the placed space
        spaces.erase(spaces.begin() + bestSpaceIdx);

        // Add right residual (width remainder x full height) and top residual (piece width x height remainder)
        if (rightResidW > 0 && sh > 0) {
            spaces.push_back({rightResidW, sh});
        }
        if (bestPw > 0 && topResidH > 0) {
            spaces.push_back({bestPw, topResidH});
        }

        // Check usability of remaining spaces: how many can still accept a piece?
        int usableSpaces = 0;
        for (auto& [w, h] : spaces) {
            if (w <= 0 || h <= 0) continue;
            for (auto [pw, ph] : allPieceDims) {
                if ((pw <= w && ph <= h) || (ph <= w && pw <= h)) {
                    usableSpaces++;
                    break;
                }
            }
        }
        double stepUsability = (double)usableSpaces / max(1, (int)spaces.size());
        totalResidualUsability += stepUsability;
    }

    if (steps == 0) return 0.0;

    double avgFill = totalFillScore / max(1, steps);
    double avgUsability = totalResidualUsability / max(1, steps);

    // Multi-step score: high fill + high residual usability across steps
    // Penalize dead-ends (spaces that become unusable quickly)
    double score = avgFill * 0.55 + avgUsability * 0.45;

    return max(0.0, min(1.0, score));
}

// Dual-gap metric: evaluates both depth and shape compatibility of gaps.
// Enhanced with aspect-ratio-weighted compatibility, orientation-specific alignment,
// spatial-aware placement pressure index, forward-looking predictive gap analysis,
// AND multi-step residual forecasting for long-term packing flexibility estimation.
static GapAnalysis computeMultiScaleGaps(const vector<int>& colH, int W,
                                          int minPieceW, int maxPieceW,
                                          double skyCV,
                                          const vector<pair<int,int>>& allPieceDims,
                                          const vector<double>& pieceOptAspects) {
    if (W <= 0) return {0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int skylineMax = 0;
    for (int h : colH) skylineMax = max(skylineMax, h);
    if (skylineMax == 0) return {0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    double cvClamped = max(0.0, min(1.5, skyCV));
    double baseDepth = 0.25 + cvClamped * 0.25;

    int activeCount = 0, passiveCount = 0;
    double totalGapArea = 0.0;
    double totalShapeCompat = 0.0;
    int totalGapsForCompat = 0;
    double totalDepthQuality = 0.0;
    double totalAspectCompat = 0.0;
    double totalOrientAlign = 0.0;
    double totalPlacementPressure = 0.0;
    double totalPredictiveScore = 0.0;
    double totalMultiStepPredictiveScore = 0.0;

    int totalDims = max(1, (int)allPieceDims.size());
    int totalPieces = max(1, (int)pieceOptAspects.size());

    for (int scale = 0; scale < 3; ++scale) {
        double depthFraction = min(0.85, baseDepth + scale * 0.15);
        int depthThreshold = (int)(skylineMax * (1.0 - depthFraction));

        int runStart = -1;
        int runMinH = INT_MAX;

        for (int i = 0; i <= W; ++i) {
            int curH = (i < W) ? colH[i] : skylineMax;
            bool inValley = (curH < depthThreshold);

            if (inValley && runStart == -1) {
                runStart = i;
                runMinH = curH;
            } else if (inValley) {
                runMinH = min(runMinH, curH);
            } else if (!inValley && runStart != -1) {
                int gapWidth = i - runStart;
                int leftH = (runStart > 0) ? colH[runStart - 1] : skylineMax;
                int rightH = (i < W) ? colH[i] : skylineMax;
                int neighborMax = max(leftH, rightH);
                int gapDepth = neighborMax - runMinH;

                totalGapArea += (double)gapWidth * gapDepth;

                double gapAspectWH = (double)gapWidth / max(1, gapDepth);
                double gapAspectHW = (double)gapDepth / max(1, gapWidth);

                int compatCount = 0;
                double aspectWeightedSum = 0.0;
                int aspectWeightedCount = 0;
                double bestOrientAlign = 0.0;

                for (int di = 0; di < (int)allPieceDims.size(); ++di) {
                    auto [pw, ph] = allPieceDims[di];
                    if (pw <= gapWidth && ph <= gapDepth) {
                        compatCount++;
                        double dimAspect = (double)pw / max(1, ph);
                        double alignScore = 1.0 / (1.0 + fabs(gapAspectWH - dimAspect));
                        aspectWeightedSum += alignScore;
                        aspectWeightedCount++;
                    }
                    if (ph <= gapWidth && pw <= gapDepth) {
                        compatCount++;
                        double dimAspectRot = (double)ph / max(1, pw);
                        double alignScore = 1.0 / (1.0 + fabs(gapAspectWH - dimAspectRot));
                        aspectWeightedSum += alignScore;
                        aspectWeightedCount++;
                    }
                }

                double compatFrac = (double)compatCount / (2.0 * totalDims);
                totalShapeCompat += compatFrac;
                totalGapsForCompat++;

                double aspectCompat = (aspectWeightedCount > 0)
                    ? aspectWeightedSum / (double)aspectWeightedCount : 0.0;
                totalAspectCompat += aspectCompat;

                double orientAlignSum = 0.0;
                for (double optAsp : pieceOptAspects) {
                    double align1 = 1.0 / (1.0 + fabs(gapAspectWH - optAsp));
                    double align2 = 1.0 / (1.0 + fabs(gapAspectHW - optAsp));
                    orientAlignSum += max(align1, align2);
                }
                double orientAlign = orientAlignSum / (double)totalPieces;
                totalOrientAlign += orientAlign;

                double depthQuality = (double)gapDepth / max(1, skylineMax);
                totalDepthQuality += depthQuality;

                double pressureIndex = computeGapPressureIndex(gapWidth, gapDepth,
                                                                leftH, rightH,
                                                                skylineMax, allPieceDims);
                totalPlacementPressure += pressureIndex;

                double predictiveScore = computePredictiveScore(gapWidth, gapDepth,
                                                                  leftH, rightH,
                                                                  skylineMax, allPieceDims);
                totalPredictiveScore += predictiveScore;

                // Multi-step residual forecasting score
                double multiStepScore = computeMultiStepPredictiveScore(gapWidth, gapDepth,
                                                                          leftH, rightH,
                                                                          skylineMax, allPieceDims);
                totalMultiStepPredictiveScore += multiStepScore;

                if (gapWidth < minPieceW) {
                    passiveCount++;
                } else if (gapWidth > maxPieceW * 3) {
                    if (gapDepth < skylineMax * 0.3) {
                        activeCount++;
                    } else {
                        passiveCount++;
                    }
                } else if (gapDepth > skylineMax * 0.7) {
                    passiveCount++;
                } else if (gapDepth < skylineMax * 0.1) {
                    passiveCount++;
                } else {
                    int heightVariation = abs(leftH - rightH);
                    if (heightVariation < skylineMax * 0.3) {
                        activeCount++;
                    } else {
                        passiveCount++;
                    }
                }

                runStart = -1;
                runMinH = INT_MAX;
            }
        }
        // Handle trailing valley
        if (runStart != -1) {
            int gapWidth = W - runStart;
            int leftH = (runStart > 0) ? colH[runStart - 1] : skylineMax;
            int gapDepth = leftH - runMinH;
            totalGapArea += (double)gapWidth * gapDepth;

            double gapAspectWH = (double)gapWidth / max(1, gapDepth);
            double gapAspectHW = (double)gapDepth / max(1, gapWidth);

            int compatCount = 0;
            double aspectWeightedSum = 0.0;
            int aspectWeightedCount = 0;
            for (int di = 0; di < (int)allPieceDims.size(); ++di) {
                auto [pw, ph] = allPieceDims[di];
                if (pw <= gapWidth && ph <= gapDepth) {
                    compatCount++;
                    double dimAspect = (double)pw / max(1, ph);
                    double alignScore = 1.0 / (1.0 + fabs(gapAspectWH - dimAspect));
                    aspectWeightedSum += alignScore;
                    aspectWeightedCount++;
                }
                if (ph <= gapWidth && pw <= gapDepth) {
                    compatCount++;
                    double dimAspectRot = (double)ph / max(1, pw);
                    double alignScore = 1.0 / (1.0 + fabs(gapAspectWH - dimAspectRot));
                    aspectWeightedSum += alignScore;
                    aspectWeightedCount++;
                }
            }

            double compatFrac = (double)compatCount / (2.0 * totalDims);
            totalShapeCompat += compatFrac;
            totalGapsForCompat++;

            double aspectCompat = (aspectWeightedCount > 0)
                ? aspectWeightedSum / (double)aspectWeightedCount : 0.0;
            totalAspectCompat += aspectCompat;

            double orientAlignSum = 0.0;
            for (double optAsp : pieceOptAspects) {
                double align1 = 1.0 / (1.0 + fabs(gapAspectWH - optAsp));
                double align2 = 1.0 / (1.0 + fabs(gapAspectHW - optAsp));
                orientAlignSum += max(align1, align2);
            }
            double orientAlign = orientAlignSum / (double)totalPieces;
            totalOrientAlign += orientAlign;

            double depthQuality = (double)gapDepth / max(1, skylineMax);
            totalDepthQuality += depthQuality;

            int rightH = skylineMax;
            double pressureIndex = computeGapPressureIndex(gapWidth, gapDepth,
                                                            leftH, rightH,
                                                            skylineMax, allPieceDims);
            totalPlacementPressure += pressureIndex;

            double predictiveScore = computePredictiveScore(gapWidth, gapDepth,
                                                              leftH, rightH,
                                                              skylineMax, allPieceDims);
            totalPredictiveScore += predictiveScore;

            double multiStepScore = computeMultiStepPredictiveScore(gapWidth, gapDepth,
                                                                      leftH, rightH,
                                                                      skylineMax, allPieceDims);
            totalMultiStepPredictiveScore += multiStepScore;

            if (gapWidth < minPieceW) {
                passiveCount++;
            } else if (gapDepth > skylineMax * 0.7 || gapDepth < skylineMax * 0.1) {
                passiveCount++;
            } else {
                activeCount++;
            }
        }
    }

    int totalGaps = activeCount + passiveCount;
    double activeRatio = (totalGaps > 0) ? (double)activeCount / totalGaps : 0.0;
    double shapeCompatScore = (totalGapsForCompat > 0) ? totalShapeCompat / totalGapsForCompat : 0.0;
    double depthScore = (totalGapsForCompat > 0) ? totalDepthQuality / totalGapsForCompat : 0.0;
    double aspectCompatScore = (totalGapsForCompat > 0) ? totalAspectCompat / totalGapsForCompat : 0.0;
    double orientationAlignScore = (totalGapsForCompat > 0) ? totalOrientAlign / totalGapsForCompat : 0.0;
    double placementPressureScore = (totalGapsForCompat > 0) ? totalPlacementPressure / totalGapsForCompat : 0.0;
    double predictiveScore = (totalGapsForCompat > 0) ? totalPredictiveScore / totalGapsForCompat : 0.0;
    double multiStepPredictiveScore = (totalGapsForCompat > 0) ? totalMultiStepPredictiveScore / totalGapsForCompat : 0.0;

    // Dual score: combine all metrics including multi-step residual forecasting
    // Weights adjusted to incorporate multiStepPredictiveScore at 0.20
    double dualScore = activeRatio * 0.15
                     + shapeCompatScore * 0.15
                     + aspectCompatScore * 0.22
                     + placementPressureScore * 0.13
                     + predictiveScore * 0.15
                     + multiStepPredictiveScore * 0.20;

    return {activeCount, passiveCount, activeRatio, totalGapArea,
            shapeCompatScore, depthScore, dualScore,
            aspectCompatScore, orientationAlignScore, placementPressureScore,
            predictiveScore, multiStepPredictiveScore};
}

// Multi-objective fragMode selection using dual-gap metrics with orientation-specific triggers
// and dynamic density-adaptive thresholds.
static int determineFragMode(const GapAnalysis& ga, double density,
                              double avgPieceAspect, double avgIrregularity,
                              double avgProtrusion) {
    int totalGaps = ga.activeCount + ga.passiveCount;
    if (totalGaps == 0) {
        return (density > 0.7) ? 2 : 0;
    }

    double aspectDeviation = fabs(avgPieceAspect - 1.0);
    double shapeComplexity = avgIrregularity + avgProtrusion;

    double activeWeight = ga.activeRatio * (1.0 + aspectDeviation * 0.6);
    double passiveWeight = (1.0 - ga.activeRatio) * (1.0 + shapeComplexity * 0.4);

    double shapeCompatBonus = ga.shapeCompatScore * 1.0;
    activeWeight += shapeCompatBonus;

    double aspectCompatBonus = ga.aspectCompatScore * 1.5;
    activeWeight += aspectCompatBonus;

    double orientAlignBonus = ga.orientationAlignScore * 2.0;
    activeWeight += orientAlignBonus;

    double pressureBonus = ga.placementPressureScore * 1.5;
    activeWeight += pressureBonus;

    double predictiveBonus = ga.predictiveScore * 1.3;
    activeWeight += predictiveBonus;

    // Multi-step residual forecasting bonus: gaps with good long-term utilization favor gap-filling
    double multiStepBonus = ga.multiStepPredictiveScore * 1.6;
    activeWeight += multiStepBonus;

    double densityAdjust = 0.0;
    if (density > 0.7) {
        densityAdjust = (density - 0.7) * 0.5;
    } else if (density < 0.5) {
        densityAdjust = -(0.5 - density) * 0.3;
    }

    double orientThreshold = 0.6 - densityAdjust * 0.15;
    double aspectThreshold = 0.4 - densityAdjust * 0.10;
    if (ga.orientationAlignScore > orientThreshold && ga.aspectCompatScore > aspectThreshold) {
        return 1;
    }

    double predictiveThreshold = 0.65 - densityAdjust * 0.10;
    if (ga.predictiveScore > predictiveThreshold && ga.aspectCompatScore > aspectThreshold) {
        return 1;
    }

    // Multi-step residual forecasting override: if long-term utilization is excellent, favor gap-filling
    double multiStepThreshold = 0.6 - densityAdjust * 0.10;
    if (ga.multiStepPredictiveScore > multiStepThreshold && ga.aspectCompatScore > aspectThreshold) {
        return 1;
    }

    double shapePenalty = (1.0 - ga.shapeCompatScore) * 0.8;
    double aspectPenalty = (1.0 - ga.aspectCompatScore) * 0.6;
    passiveWeight += shapePenalty + aspectPenalty;

    // Also penalize low multi-step utilization (dead-end gaps)
    double multiStepPenalty = (1.0 - ga.multiStepPredictiveScore) * 0.5;
    passiveWeight += multiStepPenalty;

    double dualThreshold = 0.55 - densityAdjust * 0.15;
    double activePassiveRatio = 0.9 - densityAdjust * 0.15;
    double activePrefRatio = 1.1 - densityAdjust * 0.20;
    double passivePrefRatio = 1.1 + densityAdjust * 0.20;

    if (ga.dualScore > dualThreshold && activeWeight > passiveWeight * activePassiveRatio) {
        return 1;
    }
    if (activeWeight > passiveWeight * activePrefRatio) {
        return 1;
    }
    if (passiveWeight > activeWeight * passivePrefRatio) {
        return 2;
    }
    return 0;
}

// Fixed-order packing
static bool packForWidth(int W, const vector<int>& order,
                        const vector<Piece>& pieces,
                        vector<array<int,4>>& place,
                        int& outH, double timeLimit) {
    vector<int> h(W, 0);
    int maxH = 0;
    int cnt = 0;
    for (int idx : order) {
        if (((cnt++) & 7) == 0 && elapsed() > timeLimit) return false;
        const auto& os = pieces[idx].orients;
        array<long long,5> bestKey;
        fill(bestKey.begin(), bestKey.end(), LLONG_MAX);
        int bestOid = -1, bestPx = -1, bestPy = -1;
        for (int oid = 0; oid < (int)os.size(); ++oid) {
            const Orient& o = os[oid];
            if (o.w > W) continue;
            long long area = (long long)o.w * o.h;
            for (int px = 0; px <= W - o.w; ++px) {
                int py = 0;
                for (auto [cx, cy] : o.cells) {
                    int req = h[px + cx] - cy;
                    if (req > py) py = req;
                }
                int gmax = maxH;
                long long waste = 0;
                for (auto [cx, maxCy] : o.colInfo) {
                    int top = py + maxCy + 1;
                    if (top > gmax) gmax = top;
                    waste += (long long)(top - h[px + cx]);
                }
                array<long long,5> key = {(long long)gmax, waste, (long long)py,
                                          (long long)px, area};
                if (key < bestKey) {
                    bestKey = key;
                    bestOid = oid;
                    bestPx = px;
                    bestPy = py;
                }
            }
        }
        if (bestOid == -1) return false;
        const Orient& o = os[bestOid];
        for (auto [cx, cy] : o.cells) {
            int col = bestPx + cx;
            int top = bestPy + cy + 1;
            if (top > h[col]) h[col] = top;
            if (top > maxH) maxH = top;
        }
        place[idx] = {bestPx - o.minXf, bestPy - o.minYf, o.R, o.F};
    }
    outH = maxH;
    return true;
}

// Dynamic ordering packing with bias perturbation and optional fragMode
static bool packForWidthDynamic(int W, const vector<Piece>& pieces,
                                vector<array<int,4>>& place,
                                int& outH, double timeLimit,
                                const vector<long long>& bias = {},
                                vector<int>* outOrder = nullptr,
                                int fragMode = 0) {
    int n = pieces.size();
    vector<int> h(W, 0);
    int maxH = 0;
    vector<bool> used(n, false);

    for (int step = 0; step < n; ++step) {
        if ((step & 3) == 0 && elapsed() > timeLimit) return false;

        array<long long,5> globalBestKey;
        fill(globalBestKey.begin(), globalBestKey.end(), LLONG_MAX);
        int bestIdx = -1, bestOid = -1, bestPx = -1, bestPy = -1;

        for (int pi = 0; pi < n; ++pi) {
            if (used[pi]) continue;
            const auto& os = pieces[pi].orients;
            long long b = (pi < (int)bias.size()) ? bias[pi] : 0;

            array<long long,5> pBestKey;
            fill(pBestKey.begin(), pBestKey.end(), LLONG_MAX);
            int pOid = -1, pPx = -1, pPy = -1;

            for (int oid = 0; oid < (int)os.size(); ++oid) {
                const Orient& o = os[oid];
                if (o.w > W) continue;
                long long area = (long long)o.w * o.h;
                for (int px = 0; px <= W - o.w; ++px) {
                    int py = 0;
                    for (auto [cx, cy] : o.cells) {
                        int req = h[px + cx] - cy;
                        if (req > py) py = req;
                    }
                    int gmax = maxH;
                    long long waste = 0;
                    for (auto [cx, maxCy] : o.colInfo) {
                        int top = py + maxCy + 1;
                        if (top > gmax) gmax = top;
                        waste += (long long)(top - h[px + cx]);
                    }
                    long long slope = 0;
                    int leftTop = py + o.colInfo[0].second + 1;
                    int rightTop = py + o.colInfo.back().second + 1;
                    if (px > 0) slope += abs(leftTop - h[px - 1]);
                    if (px + o.w < W) slope += abs(rightTop - h[px + o.w]);

                    array<long long,5> key = {(long long)gmax, waste, slope,
                                              (long long)py, area};
                    if (key < pBestKey) {
                        pBestKey = key;
                        pOid = oid;
                        pPx = px;
                        pPy = py;
                    }
                }
            }

            if (pOid == -1) continue;

            array<long long,5> biasedKey = pBestKey;
            biasedKey[0] += b;

            if (fragMode == 1) {
                biasedKey[0] += (long long)pieces[pi].cells.size() * 3;
            } else if (fragMode == 2) {
                biasedKey[0] -= (long long)pieces[pi].cells.size() * 2;
            }

            if (biasedKey < globalBestKey) {
                globalBestKey = biasedKey;
                bestIdx = pi;
                bestOid = pOid;
                bestPx = pPx;
                bestPy = pPy;
            }
        }

        if (bestIdx == -1) return false;
        used[bestIdx] = true;
        const Orient& o = pieces[bestIdx].orients[bestOid];
        for (auto [cx, cy] : o.cells) {
            int col = bestPx + cx;
            int top = bestPy + cy + 1;
            if (top > h[col]) h[col] = top;
            if (top > maxH) maxH = top;
        }
        place[bestIdx] = {bestPx - o.minXf, bestPy - o.minYf, o.R, o.F};
        if (outOrder) outOrder->push_back(bestIdx);
    }

    outH = maxH;
    return true;
}

// Space-aware dynamic packing with optional fragMode
static bool spaceAwarePack(int W, const vector<Piece>& pieces,
                           vector<array<int,4>>& place, int& outH,
                           mt19937& rng, double timeLimit,
                           double targetAspect, int fragMode = 0) {
    int n = (int)pieces.size();
    vector<int> h(W, 0);
    int maxH = 0;
    vector<bool> used(n, false);

    for (int step = 0; step < n; ++step) {
        if ((step & 3) == 0 && elapsed() > timeLimit) return false;

        double bestPieceScore = 1e18;
        int bestIdx = -1, bestOid = -1, bestPx = -1, bestPy = -1;

        for (int pi = 0; pi < n; ++pi) {
            if (used[pi]) continue;
            const auto& os = pieces[pi].orients;

            double pieceBestScore = 1e18;
            int pOid = -1, pPx = -1, pPy = -1;
            int validPositions = 0;
            double totalFitScore = 0;

            for (int oid = 0; oid < (int)os.size(); ++oid) {
                const Orient& o = os[oid];
                if (o.w > W) continue;

                for (int px = 0; px <= W - o.w; ++px) {
                    int py = 0;
                    for (auto [cx, cy] : o.cells) {
                        int req = h[px + cx] - cy;
                        if (req > py) py = req;
                    }
                    int gmax = maxH;
                    long long waste = 0;
                    for (auto [cx, maxCy] : o.colInfo) {
                        int top = py + maxCy + 1;
                        if (top > gmax) gmax = top;
                        waste += (long long)(top - h[px + cx]);
                    }

                    long long slope = 0;
                    int leftTop = py + o.colInfo[0].second + 1;
                    int rightTop = py + o.colInfo.back().second + 1;
                    if (px > 0) slope += abs(leftTop - h[px - 1]);
                    if (px + o.w < W) slope += abs(rightTop - h[px + o.w]);

                    validPositions++;
                    double fitScore = (double)waste + (double)slope * 0.5
                                    + (double)py * 0.3;
                    totalFitScore += fitScore;

                    double score = (double)gmax * 0.5
                                 + (double)waste * 0.6
                                 + (double)slope * 0.3
                                 + (double)py * 0.2;

                    double orientAspect = (double)o.w / max(1, o.h);
                    score += fabs(orientAspect - targetAspect) * 0.3;

                    if (score < pieceBestScore) {
                        pieceBestScore = score;
                        pOid = oid;
                        pPx = px;
                        pPy = py;
                    }
                }
            }

            if (pOid == -1) continue;

            double avgFit = (validPositions > 0) ? totalFitScore / validPositions : 1e18;
            double flexBonus = -avgFit * 0.05;

            double finalScore = pieceBestScore + flexBonus;

            if (fragMode == 1) {
                finalScore += (double)pieces[pi].cells.size() * 0.5
                            - pieces[pi].placementFlexibility * 0.3;
            } else if (fragMode == 2) {
                finalScore -= (double)pieces[pi].cells.size() * 0.5;
            }

            double rndVal = (double)(rng() % 1000) / 1000.0 - 0.5;
            finalScore += rndVal * 0.3;

            if (finalScore < bestPieceScore) {
                bestPieceScore = finalScore;
                bestIdx = pi;
                bestOid = pOid;
                bestPx = pPx;
                bestPy = pPy;
            }
        }

        if (bestIdx == -1) return false;
        used[bestIdx] = true;
        const Orient& o = pieces[bestIdx].orients[bestOid];
        for (auto [cx, cy] : o.cells) {
            int col = bestPx + cx;
            int top = bestPy + cy + 1;
            if (top > h[col]) h[col] = top;
            if (top > maxH) maxH = top;
        }
        place[bestIdx] = {bestPx - o.minXf, bestPy - o.minYf, o.R, o.F};
    }
    outH = maxH;
    return true;
}

// Multi-modal reboot packing
static bool rebootPack(int W, const vector<Piece>& pieces,
                       vector<array<int,4>>& place, int& outH,
                       mt19937& rng, double timeLimit,
                       const vector<int>& rebootOrder,
                       int mode,
                       double targetAspect) {
    if (mode == 2) {
        return spaceAwarePack(W, pieces, place, outH, rng, timeLimit, targetAspect);
    }

    int n = (int)pieces.size();
    vector<int> h(W, 0);
    int maxH = 0;

    for (int step = 0; step < n; ++step) {
        if ((step & 3) == 0 && elapsed() > timeLimit) return false;
        int idx = rebootOrder[step];
        const auto& os = pieces[idx].orients;

        double bestScore = 1e18;
        int bestOid = -1, bestPx = -1, bestPy = -1;

        for (int oid = 0; oid < (int)os.size(); ++oid) {
            const Orient& o = os[oid];
            if (o.w > W) continue;
            double aspect = max((double)o.w / max(1, o.h),
                                 (double)o.h / max(1, o.w));
            double compactness = 1.0 / max(1.0, aspect);
            double symPotential = pieces[idx].rotationalSymmetry;
            double flexPotential = pieces[idx].placementFlexibility / 8.0;

            for (int px = 0; px <= W - o.w; ++px) {
                int py = 0;
                for (auto [cx, cy] : o.cells) {
                    int req = h[px + cx] - cy;
                    if (req > py) py = req;
                }
                int gmax = maxH;
                long long waste = 0;
                for (auto [cx, maxCy] : o.colInfo) {
                    int top = py + maxCy + 1;
                    if (top > gmax) gmax = top;
                    waste += (long long)(top - h[px + cx]);
                }

                double rndVal = (double)(rng() % 1000) / 1000.0 - 0.5;
                double irrPerturb = mode == 0
                    ? pieces[idx].irregularity * 2.0 * rndVal
                    : pieces[idx].irregularity * 0.5 * rndVal;
                double aspPerturb = mode == 1
                    ? aspect * 3.0 * rndVal
                    : aspect * 1.0 * rndVal;
                double symPerturb = symPotential * 1.5 * rndVal;
                double flexPerturb = flexPotential * 1.0 * rndVal;
                double orientAspect = (double)o.w / max(1, o.h);
                double aspectMatch = -fabs(orientAspect - targetAspect) * 0.5;

                double score = (double)gmax
                             + (double)waste * 0.3
                             + (double)py * 0.5
                             + irrPerturb
                             + aspPerturb
                             + symPerturb
                             + flexPerturb
                             + aspectMatch
                             - compactness * 0.3;

                if (score < bestScore) {
                    bestScore = score;
                    bestOid = oid;
                    bestPx = px;
                    bestPy = py;
                }
            }
        }
        if (bestOid == -1) return false;
        const Orient& o = os[bestOid];
        for (auto [cx, cy] : o.cells) {
            int col = bestPx + cx;
            int top = bestPy + cy + 1;
            if (top > h[col]) h[col] = top;
            if (top > maxH) maxH = top;
        }
        place[idx] = {bestPx - o.minXf, bestPy - o.minYf, o.R, o.F};
    }
    outH = maxH;
    return true;
}

// Compute space fragmentation (skyline coefficient of variation)
static double computeFragmentation(const vector<Piece>& pieces,
                                    const vector<array<int,4>>& place, int W) {
    if (W <= 0) return 0.0;
    vector<int> colH = computeColHeights(pieces, place, W);
    double mean = 0;
    for (int h : colH) mean += h;
    mean /= W;
    if (mean < 1e-9) return 0.0;
    double var = 0;
    for (int h : colH) var += (h - mean) * (h - mean);
    var /= W;
    return sqrt(var) / mean;
}

// Compute unusable gap count: narrow valleys in skyline too small for remaining pieces
static int computeGapCount(const vector<Piece>& pieces,
                            const vector<array<int,4>>& place, int W,
                            int minPieceW, int maxH) {
    if (W <= 0 || minPieceW <= 0) return 0;
    vector<int> colH = computeColHeights(pieces, place, W);
    int skylineMax = 0;
    for (int h : colH) skylineMax = max(skylineMax, h);
    if (skylineMax == 0) return 0;
    if (maxH > 0) skylineMax = max(skylineMax, maxH);

    int gapCount = 0;
    int runStart = -1;
    double depthThreshold = skylineMax * 0.5;
    for (int i = 0; i <= W; ++i) {
        int curH = (i < W) ? colH[i] : skylineMax;
        bool inValley = (curH < depthThreshold);
        if (inValley && runStart == -1) {
            runStart = i;
        } else if (!inValley && runStart != -1) {
            int gapLen = i - runStart;
            if (gapLen < minPieceW) gapCount++;
            runStart = -1;
        }
    }
    if (runStart != -1) {
        int gapLen = W - runStart;
        if (gapLen < minPieceW) gapCount++;
    }
    return gapCount;
}

// Compute packing density
static double computeDensity(long long totalCells, int W, int H) {
    if (W <= 0 || H <= 0) return 0.0;
    return (double)totalCells / ((double)W * (double)H);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n) || n <= 0) {
        cout << "1 1\n";
        return 0;
    }

    vector<Piece> pieces(n);
    long long totalCells = 0;
    int maxAnyDim = 0;
    int maxPieceWidth = 0;
    double sumMinPieceH = 0;

    for (int i = 0; i < n; ++i) {
        int k; cin >> k;
        pieces[i].cells.resize(k);
        for (int j = 0; j < k; ++j) {
            int x, y; cin >> x >> y;
            pieces[i].cells[j] = {x, y};
        }
        pieces[i].orients = genOrients(pieces[i].cells);
        int pieceMaxW = 0, pieceMinH = INT_MAX;
        set<int> distinctWidths;
        for (const auto& o : pieces[i].orients) {
            pieces[i].minOW = min(pieces[i].minOW, o.w);
            pieces[i].maxBA = max(pieces[i].maxBA, o.w * o.h);
            pieces[i].minBA = min(pieces[i].minBA, o.w * o.h);
            pieces[i].maxAspect = max(pieces[i].maxAspect,
                                     max((double)o.w / o.h, (double)o.h / o.w));
            maxAnyDim = max(maxAnyDim, max(o.w, o.h));
            pieceMaxW = max(pieceMaxW, o.w);
            pieceMinH = min(pieceMinH, o.h);
            distinctWidths.insert(o.w);

            double asp = (double)o.w / max(1, o.h);
            pieces[i].avgAspect += asp;
            pieces[i].minAspectOverOrients = min(pieces[i].minAspectOverOrients, asp);
            pieces[i].minBW = min(pieces[i].minBW, o.w);
            pieces[i].minBH = min(pieces[i].minBH, o.h);
            if (o.w * o.h < pieces[i].bestBW * pieces[i].bestBH || pieces[i].bestBW == 0) {
                pieces[i].bestBW = o.w;
                pieces[i].bestBH = o.h;
            }
        }
        if (!pieces[i].orients.empty())
            pieces[i].avgAspect /= pieces[i].orients.size();

        double minElong = 1e18;
        for (const auto& o : pieces[i].orients) {
            double elong = max((double)o.w / max(1, o.h), (double)o.h / max(1, o.w));
            minElong = min(minElong, elong);
        }
        pieces[i].minElongation = (minElong < 1e18) ? minElong : 1.0;

        {
            double minBA_area = 1e18;
            double bestAsp = 1.0;
            double minDist = 1e18;
            double closestAsp = 1.0;
            for (const auto& o : pieces[i].orients) {
                double asp = (double)o.w / max(1, o.h);
                double area = (double)o.w * o.h;
                if (area < minBA_area) {
                    minBA_area = area;
                    bestAsp = asp;
                }
                double dist = fabs(asp - 1.0);
                if (dist < minDist) {
                    minDist = dist;
                    closestAsp = asp;
                }
            }
            pieces[i].optimalAspect = bestAsp;
            pieces[i].compactAspect = closestAsp;
        }

        maxPieceWidth = max(maxPieceWidth, pieceMaxW);
        sumMinPieceH += (pieceMinH == INT_MAX ? 0 : pieceMinH);

        pieces[i].spaceFlexibility = (double)distinctWidths.size();
        pieces[i].fittingPotential = (double)k / max(1, pieces[i].minBA);
        pieces[i].protrusionRatio = 1.0 - (double)k / max(1, pieces[i].minBA);
        pieces[i].angularDiversity = (double)pieces[i].orients.size() / 8.0;
        pieces[i].rotationalSymmetry = 1.0 - pieces[i].angularDiversity;
        pieces[i].placementFlexibility = (double)pieces[i].orients.size();

        double minAspect = 1.0;
        for (const auto& o : pieces[i].orients) {
            double asp = max((double)o.w / max(1, o.h), (double)o.h / max(1, o.w));
            minAspect = min(minAspect, asp);
        }
        pieces[i].gridAlignment = 1.0 / minAspect;

        pieces[i].conflictScore = pieces[i].protrusionRatio * 0.5
                                + (1.0 - pieces[i].angularDiversity) * 0.3
                                + pieces[i].maxAspect * 0.2;

        pieces[i].irregularity = (double)pieces[i].maxBA / max(1, k);
        pieces[i].hullArea = convexHullArea(pieces[i].cells);
        pieces[i].hullRatio = pieces[i].hullArea / max(1.0, (double)k);
        pieces[i].shapeScore = (double)pieces[i].maxBA * 0.3
                             + pieces[i].hullRatio * 20.0
                             + pieces[i].maxAspect * 10.0
                             + (double)k * 2.0;
        totalCells += k;
    }

    vector<pair<int,int>> allPieceDims;
    {
        set<pair<int,int>> dimSet;
        for (const auto& p : pieces) {
            for (const auto& o : p.orients) {
                dimSet.insert({o.w, o.h});
            }
        }
        allPieceDims.assign(dimSet.begin(), dimSet.end());
    }

    vector<double> pieceOptAspects;
    for (const auto& p : pieces) {
        pieceOptAspects.push_back(p.optimalAspect);
    }

    double avgElongation = 0;
    for (const auto& p : pieces) {
        avgElongation += p.minElongation;
    }
    avgElongation /= max(1, n);

    double cellWeightedAspect = 0, areaWeightedAspect = 0, irregWeightedAspect = 0;
    double compactWeightedAspect = 0;
    double cellWSum = 0, areaWSum = 0, irregWSum = 0, compactWSum = 0;
    for (int i = 0; i < n; ++i) {
        double cw = (double)pieces[i].cells.size();
        double aw = (double)max(1, pieces[i].minBA);
        double iw = pieces[i].irregularity + 0.1;
        double compw = pieces[i].gridAlignment + 0.1;
        cellWeightedAspect += pieces[i].optimalAspect * cw;
        cellWSum += cw;
        areaWeightedAspect += pieces[i].optimalAspect * aw;
        areaWSum += aw;
        irregWeightedAspect += pieces[i].optimalAspect * iw;
        irregWSum += iw;
        compactWeightedAspect += pieces[i].compactAspect * compw;
        compactWSum += compw;
    }
    cellWeightedAspect /= max(1.0, cellWSum);
    areaWeightedAspect /= max(1.0, areaWSum);
    irregWeightedAspect /= max(1.0, irregWSum);
    compactWeightedAspect /= max(1.0, compactWSum);

    cellWeightedAspect = max(0.5, min(2.0, cellWeightedAspect));
    areaWeightedAspect = max(0.5, min(2.0, areaWeightedAspect));
    irregWeightedAspect = max(0.5, min(2.0, irregWeightedAspect));
    compactWeightedAspect = max(0.5, min(2.0, compactWeightedAspect));

    auto [areaMinAspect, sqBalAspect, irrAspect] = multiAspectEstimate(pieces, totalCells);
    double estAspect = areaMinAspect * 0.5 + sqBalAspect * 0.3 + irrAspect * 0.2;

    auto [estW, estH] = estimateOptimalBoard(pieces, totalCells, estAspect);
    auto [sqW, sqH] = estimateOptimalBoard(pieces, totalCells, sqBalAspect);
    auto [irrW, irrH] = estimateOptimalBoard(pieces, totalCells, irrAspect);

    double avgPieceH = sumMinPieceH / max(1, n);
    int criticalWidth = maxPieceWidth + (int)round(avgPieceH);

    vector<array<int,4>> bestPlace(n);
    int bestW = 0, bestH = 0;
    long long bestArea = LLONG_MAX;
    {
        int curX = 0, boardH = 0;
        for (int i = 0; i < n; ++i) {
            const Orient* bo = nullptr;
            long long ba = LLONG_MAX;
            for (const auto& o : pieces[i].orients) {
                long long a = (long long)o.w * o.h;
                if (a < ba || (bo && a == ba && o.h < bo->h) ||
                    (bo && a == ba && o.h == bo->h && o.w < bo->w)) {
                    ba = a; bo = &o;
                }
            }
            const Orient& o = *bo;
            int X = curX - o.minXf;
            int Y = -o.minYf;
            bestPlace[i] = {X, Y, o.R, o.F};
            curX += o.w;
            boardH = max(boardH, o.h);
        }
        bestW = curX;
        bestH = boardH;
        bestArea = (long long)bestW * bestH;
    }

    int maxminOW = 0;
    for (const auto& p : pieces) maxminOW = max(maxminOW, p.minOW);

    criticalWidth = max(criticalWidth, maxminOW);
    criticalWidth = max(criticalWidth, estW);
    criticalWidth = max(criticalWidth, sqW);
    criticalWidth = max(criticalWidth, irrW);

    int baseW = max(maxminOW, (int)ceil(sqrt((double)totalCells)));
    baseW = max(baseW, estW);
    long long ubWll = min(totalCells, (long long)max(baseW * 2, baseW + 20));
    ubWll = max(ubWll, (long long)criticalWidth + 10);
    int ubW = (int)min<long long>(ubWll, 4000);
    ubW = max(ubW, criticalWidth);

    vector<vector<int>> pieceWidths(n);
    for (int i = 0; i < n; ++i) {
        for (const auto& o : pieces[i].orients) {
            pieceWidths[i].push_back(o.w);
        }
    }

    double instanceChallenge = 0;
    for (int i = 0; i < n; ++i) {
        instanceChallenge += pieces[i].protrusionRatio
                            + min(pieces[i].maxAspect, 10.0) * 0.1;
    }
    instanceChallenge /= max(1, n);

    vector<double> pressureTable(ubW + 2, 0.0);
    {
        vector<double> pressureDiff(ubW + 2, 0.0);
        for (int i = 0; i < n; ++i) {
            int total = (int)pieceWidths[i].size();
            if (total == 0) continue;

            double weight = 1.0 + pieces[i].protrusionRatio
                          + min(pieces[i].maxAspect, 10.0) * 0.1;
            double decrement = weight / total;

            int widthCnt[16] = {0};
            for (int w : pieceWidths[i]) {
                if (w >= 0 && w < 16) widthCnt[w]++;
            }

            pressureDiff[0] += weight;
            for (int w = 0; w < 16; ++w) {
                if (widthCnt[w] > 0 && w <= ubW) {
                    pressureDiff[w] -= decrement * widthCnt[w];
                }
            }
        }

        double acc = 0;
        for (int W = 0; W <= ubW; ++W) {
            acc += pressureDiff[W];
            pressureTable[W] = acc / max(1, n);
        }
    }

    auto placementPressure = [&](int W) -> double {
        if (W < 0 || W >= (int)pressureTable.size()) return 0.0;
        return pressureTable[W];
    };

    vector<int> orderBA(n), orderIrr(n), orderShape(n), orderFit(n), orderSym(n);
    iota(orderBA.begin(), orderBA.end(), 0);
    iota(orderIrr.begin(), orderIrr.end(), 0);
    iota(orderShape.begin(), orderShape.end(), 0);
    iota(orderFit.begin(), orderFit.end(), 0);
    iota(orderSym.begin(), orderSym.end(), 0);
    sort(orderBA.begin(), orderBA.end(), [&](int a, int b) {
        if (pieces[a].maxBA != pieces[b].maxBA) return pieces[a].maxBA > pieces[b].maxBA;
        return pieces[a].cells.size() > pieces[b].cells.size();
    });
    sort(orderIrr.begin(), orderIrr.end(), [&](int a, int b) {
        if (pieces[a].irregularity != pieces[b].irregularity)
            return pieces[a].irregularity > pieces[b].irregularity;
        if (pieces[a].maxAspect != pieces[b].maxAspect)
            return pieces[a].maxAspect > pieces[b].maxAspect;
        return pieces[a].cells.size() > pieces[b].cells.size();
    });
    sort(orderShape.begin(), orderShape.end(), [&](int a, int b) {
        if (pieces[a].shapeScore != pieces[b].shapeScore)
            return pieces[a].shapeScore > pieces[b].shapeScore;
        if (pieces[a].hullRatio != pieces[b].hullRatio)
            return pieces[a].hullRatio > pieces[b].hullRatio;
        return pieces[a].cells.size() > pieces[b].cells.size();
    });
    sort(orderFit.begin(), orderFit.end(), [&](int a, int b) {
        if (pieces[a].fittingPotential != pieces[b].fittingPotential)
            return pieces[a].fittingPotential > pieces[b].fittingPotential;
        if (pieces[a].minBA != pieces[b].minBA)
            return pieces[a].minBA < pieces[b].minBA;
        return pieces[a].cells.size() > pieces[b].cells.size();
    });
    sort(orderSym.begin(), orderSym.end(), [&](int a, int b) {
        if (pieces[a].rotationalSymmetry != pieces[b].rotationalSymmetry)
            return pieces[a].rotationalSymmetry > pieces[b].rotationalSymmetry;
        if (pieces[a].maxBA != pieces[b].maxBA)
            return pieces[a].maxBA > pieces[b].maxBA;
        return pieces[a].cells.size() > pieces[b].cells.size();
    });

    vector<int> orderSpace(n);
    iota(orderSpace.begin(), orderSpace.end(), 0);
    sort(orderSpace.begin(), orderSpace.end(), [&](int a, int b) {
        if (pieces[a].spaceFlexibility != pieces[b].spaceFlexibility)
            return pieces[a].spaceFlexibility > pieces[b].spaceFlexibility;
        if (pieces[a].placementFlexibility != pieces[b].placementFlexibility)
            return pieces[a].placementFlexibility > pieces[b].placementFlexibility;
        return pieces[a].cells.size() > pieces[b].cells.size();
    });

    vector<int> rebootOrderIrr(n), rebootOrderGrid(n), rebootOrderSpace(n);
    iota(rebootOrderIrr.begin(), rebootOrderIrr.end(), 0);
    iota(rebootOrderGrid.begin(), rebootOrderGrid.end(), 0);
    iota(rebootOrderSpace.begin(), rebootOrderSpace.end(), 0);
    sort(rebootOrderIrr.begin(), rebootOrderIrr.end(), [&](int a, int b) {
        double sa = pieces[a].irregularity + pieces[a].protrusionRatio;
        double sb = pieces[b].irregularity + pieces[b].protrusionRatio;
        if (sa != sb) return sa > sb;
        return pieces[a].cells.size() > pieces[b].cells.size();
    });
    sort(rebootOrderGrid.begin(), rebootOrderGrid.end(), [&](int a, int b) {
        if (pieces[a].gridAlignment != pieces[b].gridAlignment)
            return pieces[a].gridAlignment > pieces[b].gridAlignment;
        if (pieces[a].minBA != pieces[b].minBA)
            return pieces[a].minBA > pieces[b].minBA;
        return pieces[a].cells.size() > pieces[b].cells.size();
    });
    sort(rebootOrderSpace.begin(), rebootOrderSpace.end(), [&](int a, int b) {
        double sa = pieces[a].spaceFlexibility + pieces[a].placementFlexibility / 8.0;
        double sb = pieces[b].spaceFlexibility + pieces[b].placementFlexibility / 8.0;
        if (sa != sb) return sa > sb;
        return pieces[a].cells.size() > pieces[b].cells.size();
    });

    double avgPieceAspectStat = 0, avgIrregStat = 0, avgProtrusionStat = 0;
    for (const auto& p : pieces) {
        avgPieceAspectStat += p.optimalAspect;
        avgIrregStat += p.irregularity;
        avgProtrusionStat += p.protrusionRatio;
    }
    avgPieceAspectStat /= max(1, n);
    avgIrregStat /= max(1, n);
    avgProtrusionStat /= max(1, n);

    mt19937 rng(42);
    double mu1 = sqrt((double)totalCells);
    double mu2 = (double)criticalWidth;
    double sigma1 = max(1.0, mu1 * 0.2);
    double sigma2 = max(1.0, mu2 * 0.15);
    normal_distribution<double> gaussDist1(mu1, sigma1);
    normal_distribution<double> gaussDist2(mu2, sigma2);

    set<int> stage1Widths;
    if (maxAnyDim >= maxminOW) {
        for (int d = 0; d <= 12; ++d) {
            int w = maxAnyDim + d;
            if (w >= maxminOW && w <= ubW) stage1Widths.insert(w);
        }
    }
    for (int mult = 1; mult <= 4; ++mult) {
        int w = maxAnyDim * mult;
        if (w >= maxminOW && w <= ubW) stage1Widths.insert(w);
        w = (maxAnyDim * mult + 1) / 2;
        if (w >= maxminOW && w <= ubW) stage1Widths.insert(w);
    }
    for (int d = -3; d <= 3; ++d) {
        int w = baseW + d;
        if (w >= maxminOW && w <= ubW) stage1Widths.insert(w);
    }
    for (int d = -6; d <= 6; ++d) {
        int w = criticalWidth + d;
        if (w >= maxminOW && w <= ubW) stage1Widths.insert(w);
    }
    for (int d = -5; d <= 5; ++d) {
        int w = estW + d;
        if (w >= maxminOW && w <= ubW) stage1Widths.insert(w);
    }
    for (int d = -4; d <= 4; ++d) {
        int w = sqW + d;
        if (w >= maxminOW && w <= ubW) stage1Widths.insert(w);
    }
    for (int d = -4; d <= 4; ++d) {
        int w = irrW + d;
        if (w >= maxminOW && w <= ubW) stage1Widths.insert(w);
    }

    set<int> stage2Widths;
    for (int i = 0; i < 30; ++i) {
        int w = (int)round(gaussDist1(rng));
        if (w >= maxminOW && w <= ubW) stage2Widths.insert(w);
    }
    for (int i = 0; i < 30; ++i) {
        int w = (int)round(gaussDist2(rng));
        if (w >= maxminOW && w <= ubW) stage2Widths.insert(w);
    }
    for (int d = -5; d <= 5; ++d) {
        int w = baseW + d;
        if (w >= maxminOW && w <= ubW) stage2Widths.insert(w);
    }
    for (int d = -3; d <= 3; ++d) {
        int w = criticalWidth + d;
        if (w >= maxminOW && w <= ubW) stage2Widths.insert(w);
    }
    for (int d = -3; d <= 3; ++d) {
        int w = estW + d;
        if (w >= maxminOW && w <= ubW) stage2Widths.insert(w);
    }
    for (int d = -3; d <= 3; ++d) {
        int w = sqW + d;
        if (w >= maxminOW && w <= ubW) stage2Widths.insert(w);
    }
    for (int d = -3; d <= 3; ++d) {
        int w = irrW + d;
        if (w >= maxminOW && w <= ubW) stage2Widths.insert(w);
    }

    vector<int> stage1Ws(stage1Widths.begin(), stage1Widths.end());
    vector<int> stage2Ws(stage2Widths.begin(), stage2Widths.end());

    set<int> allWidthSet = stage1Widths;
    allWidthSet.insert(stage2Widths.begin(), stage2Widths.end());
    vector<int> Ws(allWidthSet.begin(), allWidthSet.end());
    sort(Ws.begin(), Ws.end());

    int bestWForRefine = -1;

    int boostedAspect = -1;
    int boostRemaining = 0;
    double maxObservedDensity = 0.0;

    double adaptiveTargetAspect = estAspect;
    TripleEMA densityEMA;

    auto updateAdaptiveAspect = [&](int W, int H, bool isNewBest) {
        if (W > 0 && H > 0) {
            double observedAspect = (double)W / (double)H;
            double density = computeDensity(totalCells, W, H);
            maxObservedDensity = max(maxObservedDensity, density);

            if (isNewBest) {
                double devArea = fabs(observedAspect - areaMinAspect);
                double devSq = fabs(observedAspect - sqBalAspect);
                double devIrr = fabs(observedAspect - irrAspect);
                if (devArea <= devSq && devArea <= devIrr) boostedAspect = 0;
                else if (devSq <= devIrr) boostedAspect = 1;
                else boostedAspect = 2;
                boostRemaining = 5;
            }
            if (boostRemaining > 0) boostRemaining--;

            densityEMA.update(density);
            double tem = densityEMA.value();

            double wArea = 0.25 + tem * 0.45;
            double wSquare = 0.15 + (1.0 - tem) * 0.35;
            double wIrr = 1.0 - wArea - wSquare;
            wIrr = max(0.05, wIrr);

            if (boostRemaining > 0 && boostedAspect >= 0) {
                double boostMult = 1.2;
                if (boostedAspect == 0) wArea *= boostMult;
                else if (boostedAspect == 1) wSquare *= boostMult;
                else wIrr *= boostMult;
            }

            double wsum = wArea + wSquare + wIrr;
            wArea /= wsum; wSquare /= wsum; wIrr /= wsum;
            double compositeAspect = wArea * areaMinAspect
                                   + wSquare * sqBalAspect
                                   + wIrr * irrAspect;
            double weight = density * 0.35;
            adaptiveTargetAspect = compositeAspect * (1.0 - weight)
                                 + observedAspect * weight;
            adaptiveTargetAspect = max(0.5, min(2.0, adaptiveTargetAspect));
        }
    };

    auto updateFragModeFromGaps = [&](int& fragMode, const vector<array<int,4>>& place,
                                       int W, int H) {
        vector<int> colH = computeColHeights(pieces, place, W);
        double meanH = 0;
        for (int h : colH) meanH += h;
        meanH /= max(1, W);
        double varH = 0;
        for (int h : colH) varH += (h - meanH) * (h - meanH);
        varH /= max(1, W);
        double skyCV = (meanH > 1e-9) ? sqrt(varH) / meanH : 0.0;

        GapAnalysis ga = computeMultiScaleGaps(colH, W, maxminOW, maxPieceWidth,
                                                skyCV, allPieceDims, pieceOptAspects);
        double density = computeDensity(totalCells, W, H);
        fragMode = determineFragMode(ga, density, avgPieceAspectStat,
                                     avgIrregStat, avgProtrusionStat);
    };

    // Phase 1a
    double sweepTime1 = 0.25;
    for (int W : stage1Ws) {
        if (elapsed() > sweepTime1) break;
        for (int oi = 0; oi < 5; ++oi) {
            if (elapsed() > sweepTime1) break;
            const vector<int>& ord = (oi == 0) ? orderShape :
                                     (oi == 1) ? orderBA :
                                     (oi == 2) ? orderIrr :
                                     (oi == 3) ? orderFit : orderSym;
            vector<array<int,4>> place(n);
            int H;
            if (packForWidth(W, ord, pieces, place, H, sweepTime1)) {
                long long area = (long long)W * H;
                bool isNewBest = area < bestArea ||
                    (area == bestArea && H < bestH) ||
                    (area == bestArea && H == bestH && W < bestW);
                if (isNewBest) {
                    bestArea = area;
                    bestH = H;
                    bestW = W;
                    bestPlace = place;
                    bestWForRefine = W;
                }
                updateAdaptiveAspect(W, H, isNewBest);
            }
        }
    }

    // Phase 1b
    double sweepTime2 = 0.5;
    for (int W : stage2Ws) {
        if (elapsed() > sweepTime2) break;
        for (int oi = 0; oi < 5; ++oi) {
            if (elapsed() > sweepTime2) break;
            const vector<int>& ord = (oi == 0) ? orderShape :
                                     (oi == 1) ? orderBA :
                                     (oi == 2) ? orderIrr :
                                     (oi == 3) ? orderFit : orderSym;
            vector<array<int,4>> place(n);
            int H;
            if (packForWidth(W, ord, pieces, place, H, sweepTime2)) {
                long long area = (long long)W * H;
                bool isNewBest = area < bestArea ||
                    (area == bestArea && H < bestH) ||
                    (area == bestArea && H == bestH && W < bestW);
                if (isNewBest) {
                    bestArea = area;
                    bestH = H;
                    bestW = W;
                    bestPlace = place;
                    bestWForRefine = W;
                }
                updateAdaptiveAspect(W, H, isNewBest);
            }
        }
    }

    // Phase 2: Dynamic packing
    double dynTime = 0.95;

    vector<int> dynWs;
    if (bestWForRefine != -1) {
        for (int d = -4; d <= 4; ++d) {
            int w = bestWForRefine + d;
            if (w >= maxminOW && w <= ubW) dynWs.push_back(w);
        }
    }
    for (int d = -3; d <= 3; ++d) {
        int w = criticalWidth + d;
        if (w >= maxminOW && w <= ubW) dynWs.push_back(w);
    }
    for (int d = -3; d <= 3; ++d) {
        int w = estW + d;
        if (w >= maxminOW && w <= ubW) dynWs.push_back(w);
    }
    for (int d = -3; d <= 3; ++d) {
        int w = irrW + d;
        if (w >= maxminOW && w <= ubW) dynWs.push_back(w);
    }
    {
        int adW = max(1, (int)round(sqrt((double)totalCells) * sqrt(adaptiveTargetAspect)));
        for (int d = -3; d <= 3; ++d) {
            int w = adW + d;
            if (w >= maxminOW && w <= ubW) dynWs.push_back(w);
        }
    }
    for (int W : Ws) {
        if (find(dynWs.begin(), dynWs.end(), W) == dynWs.end())
            dynWs.push_back(W);
    }
    sort(dynWs.begin(), dynWs.end());
    dynWs.erase(unique(dynWs.begin(), dynWs.end()), dynWs.end());

    sort(dynWs.begin(), dynWs.end(), [&](int a, int b) {
        double pa = placementPressure(a);
        double pb = placementPressure(b);
        double sa = (double)a + 1.0 * pa;
        double sb = (double)b + 1.0 * pb;
        if (fabs(sa - sb) > 0.01) return sa < sb;
        return a < b;
    });

    int curFragMode = 0;

    for (int W : dynWs) {
        if (elapsed() > dynTime) break;
        vector<array<int,4>> place(n);
        int H;
        int useFragMode = curFragMode;
        if (packForWidthDynamic(W, pieces, place, H, dynTime, {}, nullptr, useFragMode)) {
            long long area = (long long)W * H;
            bool isNewBest = area < bestArea ||
                (area == bestArea && H < bestH) ||
                (area == bestArea && H == bestH && W < bestW);
            if (isNewBest) {
                bestArea = area;
                bestH = H;
                bestW = W;
                bestPlace = place;
                bestWForRefine = W;
            }
            updateAdaptiveAspect(W, H, isNewBest);

            updateFragModeFromGaps(curFragMode, place, W, H);
        }
    }

    // Phase 2b: Space-aware packing
    double spaceTime = 1.15;
    {
        vector<int> spaceWs;
        if (bestWForRefine != -1) {
            for (int d = -3; d <= 3; ++d) {
                int w = bestWForRefine + d;
                if (w >= maxminOW && w <= ubW) spaceWs.push_back(w);
            }
        }
        for (int d = -2; d <= 2; ++d) {
            int w = criticalWidth + d;
            if (w >= maxminOW && w <= ubW) spaceWs.push_back(w);
        }
        for (int d = -2; d <= 2; ++d) {
            int w = estW + d;
            if (w >= maxminOW && w <= ubW) spaceWs.push_back(w);
        }
        for (int d = -2; d <= 2; ++d) {
            int w = irrW + d;
            if (w >= maxminOW && w <= ubW) spaceWs.push_back(w);
        }
        {
            int adW = max(1, (int)round(sqrt((double)totalCells) * sqrt(adaptiveTargetAspect)));
            for (int d = -2; d <= 2; ++d) {
                int w = adW + d;
                if (w >= maxminOW && w <= ubW) spaceWs.push_back(w);
            }
        }
        for (int W : Ws) {
            if (find(spaceWs.begin(), spaceWs.end(), W) == spaceWs.end())
                spaceWs.push_back(W);
        }
        sort(spaceWs.begin(), spaceWs.end());
        spaceWs.erase(unique(spaceWs.begin(), spaceWs.end()), spaceWs.end());
        sort(spaceWs.begin(), spaceWs.end(), [&](int a, int b) {
            double pa = placementPressure(a);
            double pb = placementPressure(b);
            double sa = (double)a + 1.0 * pa;
            double sb = (double)b + 1.0 * pb;
            if (fabs(sa - sb) > 0.01) return sa < sb;
            return a < b;
        });

        double targetAsp = (bestH > 0)
            ? ((double)bestW / (double)bestH) * 0.5 + adaptiveTargetAspect * 0.5
            : adaptiveTargetAspect;

        int spaceFragMode = curFragMode;

        for (int W : spaceWs) {
            if (elapsed() > spaceTime) break;
            vector<array<int,4>> place(n);
            int H;
            if (spaceAwarePack(W, pieces, place, H, rng, spaceTime, targetAsp, spaceFragMode)) {
                long long area = (long long)W * H;
                bool isNewBest = area < bestArea ||
                    (area == bestArea && H < bestH) ||
                    (area == bestArea && H == bestH && W < bestW);
                if (isNewBest) {
                    bestArea = area;
                    bestH = H;
                    bestW = W;
                    bestPlace = place;
                    bestWForRefine = W;
                }
                updateAdaptiveAspect(W, H, isNewBest);

                updateFragModeFromGaps(spaceFragMode, place, W, H);
            }
        }
    }

    // Phase 2c: Dynamic aspect ratio explorer
    double mutateTime = 1.4;
    if (elapsed() < mutateTime) {
        double bestRatio = (bestH > 0) ? (double)bestW / (double)bestH : estAspect;

        vector<double> mutatedAspects;

        for (int w = 0; w <= 10; ++w) {
            double blendW = (double)w / 10.0;
            mutatedAspects.push_back(bestRatio * blendW + cellWeightedAspect * (1.0 - blendW));
        }
        for (int w = 0; w <= 10; ++w) {
            double blendW = (double)w / 10.0;
            mutatedAspects.push_back(bestRatio * blendW + areaWeightedAspect * (1.0 - blendW));
        }
        for (int w = 0; w <= 10; ++w) {
            double blendW = (double)w / 10.0;
            mutatedAspects.push_back(bestRatio * blendW + irregWeightedAspect * (1.0 - blendW));
        }
        for (int w = 0; w <= 6; ++w) {
            double blendW = (double)w / 6.0;
            mutatedAspects.push_back(bestRatio * blendW + compactWeightedAspect * (1.0 - blendW));
        }
        for (int w = 1; w <= 5; ++w) {
            double blendW = (double)w / 6.0;
            mutatedAspects.push_back(bestRatio * blendW + estAspect * (1.0 - blendW));
        }
        for (int i = 0; i < 15; ++i) {
            int which = rng() % 4;
            double scale;
            if (which == 0) scale = cellWeightedAspect;
            else if (which == 1) scale = areaWeightedAspect;
            else if (which == 2) scale = irregWeightedAspect;
            else scale = compactWeightedAspect;
            double noise = ((double)(rng() % 1000) / 1000.0 - 0.5) * 0.4 * scale;
            double mutated = bestRatio + noise;
            mutatedAspects.push_back(max(0.5, min(2.0, mutated)));
        }
        for (int i = 0; i < 8; ++i) {
            double noise = ((double)(rng() % 1000) / 1000.0 - 0.5) * 0.3;
            double mutated = cellWeightedAspect + noise;
            mutatedAspects.push_back(max(0.5, min(2.0, mutated)));
        }
        for (int i = 0; i < 8; ++i) {
            double noise = ((double)(rng() % 1000) / 1000.0 - 0.5) * 0.3;
            double mutated = areaWeightedAspect + noise;
            mutatedAspects.push_back(max(0.5, min(2.0, mutated)));
        }
        for (int i = 0; i < 8; ++i) {
            double noise = ((double)(rng() % 1000) / 1000.0 - 0.5) * 0.3;
            double mutated = irregWeightedAspect + noise;
            mutatedAspects.push_back(max(0.5, min(2.0, mutated)));
        }
        for (int i = 0; i < 8; ++i) {
            double noise = ((double)(rng() % 1000) / 1000.0 - 0.5) * 0.3;
            double mutated = compactWeightedAspect + noise;
            mutatedAspects.push_back(max(0.5, min(2.0, mutated)));
        }
        mutatedAspects.push_back(bestRatio * 0.5 + areaMinAspect * 0.5);
        mutatedAspects.push_back(bestRatio * 0.5 + sqBalAspect * 0.5);
        mutatedAspects.push_back(bestRatio * 0.5 + irrAspect * 0.5);
        mutatedAspects.push_back(bestRatio * 0.5 + cellWeightedAspect * 0.5);
        mutatedAspects.push_back(bestRatio * 0.5 + areaWeightedAspect * 0.5);
        mutatedAspects.push_back(bestRatio * 0.5 + irregWeightedAspect * 0.5);
        mutatedAspects.push_back(bestRatio * 0.5 + compactWeightedAspect * 0.5);
        mutatedAspects.push_back(cellWeightedAspect * 0.5 + compactWeightedAspect * 0.5);
        mutatedAspects.push_back(areaWeightedAspect * 0.5 + compactWeightedAspect * 0.5);
        mutatedAspects.push_back(irregWeightedAspect * 0.5 + compactWeightedAspect * 0.5);

        sort(mutatedAspects.begin(), mutatedAspects.end());
        mutatedAspects.erase(unique(mutatedAspects.begin(), mutatedAspects.end(),
            [](double a, double b) { return fabs(a - b) < 0.01; }),
            mutatedAspects.end());

        int mutateFragMode = curFragMode;

        for (double asp : mutatedAspects) {
            if (elapsed() > mutateTime) break;
            int w = max(1, (int)round(sqrt((double)totalCells) * sqrt(asp)));
            for (int d = -2; d <= 2; ++d) {
                if (elapsed() > mutateTime) break;
                int W = w + d;
                if (W < maxminOW || W > ubW) continue;

                {
                    vector<array<int,4>> place(n);
                    int H;
                    if (packForWidthDynamic(W, pieces, place, H, mutateTime, {}, nullptr, mutateFragMode)) {
                        long long area = (long long)W * H;
                        bool isNewBest = area < bestArea ||
                            (area == bestArea && H < bestH) ||
                            (area == bestArea && H == bestH && W < bestW);
                        if (isNewBest) {
                            bestArea = area;
                            bestH = H;
                            bestW = W;
                            bestPlace = place;
                            bestWForRefine = W;
                        }
                        updateAdaptiveAspect(W, H, isNewBest);

                        updateFragModeFromGaps(mutateFragMode, place, W, H);
                    }
                }
                if (elapsed() > mutateTime) break;
                {
                    vector<array<int,4>> place(n);
                    int H;
                    if (spaceAwarePack(W, pieces, place, H, rng, mutateTime, asp, mutateFragMode)) {
                        long long area = (long long)W * H;
                        bool isNewBest = area < bestArea ||
                            (area == bestArea && H < bestH) ||
                            (area == bestArea && H == bestH && W < bestW);
                        if (isNewBest) {
                            bestArea = area;
                            bestH = H;
                            bestW = W;
                            bestPlace = place;
                            bestWForRefine = W;
                        }
                        updateAdaptiveAspect(W, H, isNewBest);

                        updateFragModeFromGaps(mutateFragMode, place, W, H);
                    }
                }
            }
        }
    }

    // Phase 3: Tri-phase SA with global reconfiguration pulses and aspect mutation
    double saTime = 1.9;
    if (bestWForRefine != -1 && elapsed() < saTime) {
        vector<double> conflictScore(n);
        double totalConflict = 0;
        for (int i = 0; i < n; ++i) {
            conflictScore[i] = pieces[i].conflictScore;
            totalConflict += max(0.01, conflictScore[i]);
        }
        vector<double> cumConflict(n);
        {
            double acc = 0;
            for (int i = 0; i < n; ++i) {
                acc += max(0.01, conflictScore[i]);
                cumConflict[i] = acc;
            }
        }
        auto pickHighConflictPiece = [&]() -> int {
            if (totalConflict <= 0) return rng() % n;
            double r = (double)(rng() % 100000) / 100000.0 * totalConflict;
            int lo = 0, hi = n - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (cumConflict[mid] < r) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        };

        auto dynPress = [&](int candW, double curArea, int curW, int curH,
                             double curFrag) -> double {
            double base = placementPressure(candW);
            double predH = max(1.0, (double)curArea / max(1, candW));
            double aspect = max((double)candW, predH) / max(1.0, min((double)candW, predH));
            double aspectPressure = max(0.0, aspect - 1.5) * 0.5;
            double widthDeltaRatio = (curW > 0) ? (double)abs(candW - curW) / curW : 0.0;
            double fragPressure = curFrag * (1.0 + widthDeltaRatio * 0.1);
            double boardAspect = (double)candW / predH;
            double aspectDev = fabs(boardAspect - adaptiveTargetAspect);
            double sqDev = fabs(boardAspect - sqBalAspect);
            double irrDev = fabs(boardAspect - irrAspect);

            double curDensity = (curW > 0 && curH > 0) ? (double)totalCells / ((double)curW * (double)curH) : 0.0;
            double irrPenaltyMult = (curDensity > 0.75) ? 0.5 : 1.0;

            double multiAspectPressure = (aspectDev + sqDev + irrDev * irrPenaltyMult) * 0.15;
            return base + aspectPressure + fragPressure + multiAspectPressure;
        };

        auto aspectBiasedWidthSample = [&](int curW, int curH,
                                            double bestAspectRatio) -> int {
            double predH = max(1.0, (double)totalCells / max(1, curW));
            double curAspect = (double)curW / max(1.0, predH);
            double gradDir = (bestAspectRatio - curAspect);
            int maxD = 6;
            int step = 0;
            if (gradDir > 0.1) {
                step = 1 + (int)(rng() % maxD);
            } else if (gradDir < -0.1) {
                step = -(1 + (int)(rng() % maxD));
            } else {
                step = (int)(rng() % (2 * maxD + 1)) - maxD;
            }
            step += (int)((rng() % 5) - 2);
            step = max(-maxD, min(maxD, step));
            int candW = max(maxminOW, min(ubW, curW + step));
            return candW;
        };

        auto contextualAspectScore = [&](double density, double frag) -> double {
            double compactWeight;
            if (density > 0.7) {
                compactWeight = 0.3 + (density - 0.7) * 2.0;
            } else if (density < 0.5) {
                compactWeight = max(0.05, 0.15 - (0.5 - density) * 0.4);
            } else {
                compactWeight = 0.2;
            }
            compactWeight += frag * 0.15;
            compactWeight = max(0.05, min(0.8, compactWeight));

            double elongWeight = 1.0 - compactWeight;
            elongWeight = max(0.05, elongWeight);

            double wsum = compactWeight + elongWeight;
            compactWeight /= wsum;
            elongWeight /= wsum;

            double compactMetric = compactWeightedAspect * 0.6 + sqBalAspect * 0.4;
            double elongMetric = cellWeightedAspect * 0.5 + irregWeightedAspect * 0.5;

            return compactWeight * compactMetric + elongWeight * elongMetric;
        };

        auto mutateAspect = [&](double bestRatio, double density, double frag,
                                int gapCount, int minPieceW) -> double {
            double contextual = contextualAspectScore(density, frag);

            double gapNorm = (minPieceW > 0) ? (double)gapCount / (double)minPieceW : 0.0;
            double gapThreshold = 0.3 + density * 0.4;
            if (gapNorm > gapThreshold) {
                contextual = contextual * 0.4
                           + compactWeightedAspect * 0.4
                           + sqBalAspect * 0.2;
            }

            double blendW = (double)(rng() % 100) / 100.0;
            double mutated = bestRatio * blendW + contextual * (1.0 - blendW);

            if (rng() % 3 == 0) {
                int whichMetric = rng() % 4;
                double pieceMetric;
                if (whichMetric == 0) pieceMetric = cellWeightedAspect;
                else if (whichMetric == 1) pieceMetric = areaWeightedAspect;
                else if (whichMetric == 2) pieceMetric = irregWeightedAspect;
                else pieceMetric = compactWeightedAspect;
                if (density > 0.7 && gapNorm > gapThreshold) {
                    pieceMetric = compactWeightedAspect;
                } else if (density > 0.7 && whichMetric == 3) {
                    pieceMetric = compactWeightedAspect;
                } else if (density < 0.5 && whichMetric <= 1) {
                    pieceMetric = (whichMetric == 0) ? cellWeightedAspect : irregWeightedAspect;
                }
                mutated = mutated * 0.5 + pieceMetric * 0.5;
            }

            double noise = ((double)(rng() % 1000) / 1000.0 - 0.5) * 0.3 * contextual;
            mutated += noise;
            return max(0.5, min(2.0, mutated));
        };

        vector<long long> curBias(n, 0);
        int curW = bestWForRefine;
        int curH = bestH;
        vector<array<int,4>> curPlace = bestPlace;
        long long curArea = bestArea;
        vector<int> curOrder;

        double curFragmentation = computeFragmentation(pieces, curPlace, curW);
        double curDensity = computeDensity(totalCells, curW, curH);
        int curGapCount = computeGapCount(pieces, curPlace, curW, maxminOW, curH);

        double bestAspectRatio = (bestH > 0) ? (double)bestW / (double)bestH : estAspect;

        double saStart = elapsed();
        double saDuration = saTime - saStart;
        double phaseBoundary = 0.5;

        double T0_wide = max(1.0, (double)curArea * 0.08);
        double T0_narrow = max(0.5, (double)curArea * 0.03);
        double Tmin = 0.001;
        int stagnation = 0;

        int aspectMutationCounter = 0;
        int aspectMutationInterval = 50;

        int phase1BestW = curW;
        int phase1BestH = curH;
        long long phase1BestArea = curArea;
        vector<array<int,4>> phase1BestPlace = curPlace;

        bool phaseTransitioned = false;
        bool inReboot = false;
        double rebootEndTime = 0.0;
        int rebootMode = 0;

        int saFragMode = 0;

        while (elapsed() < saTime) {
            if (inReboot) {
                if (elapsed() >= rebootEndTime) {
                    inReboot = false;
                }
            }

            vector<long long> newBias = curBias;
            int tryW = curW;

            double frac = max(0.0, min(1.0, (elapsed() - saStart) / saDuration));
            bool widePhase = (frac < phaseBoundary);

            aspectMutationCounter++;
            if (aspectMutationCounter >= aspectMutationInterval) {
                aspectMutationCounter = 0;
                aspectMutationInterval = 30 + rng() % 40;

                curDensity = computeDensity(totalCells, curW, curH);
                curFragmentation = computeFragmentation(pieces, curPlace, curW);
                curGapCount = computeGapCount(pieces, curPlace, curW, maxminOW, curH);

                double curBestRatio = (bestH > 0) ? (double)bestW / (double)bestH : estAspect;
                double mutatedAsp = mutateAspect(curBestRatio, curDensity, curFragmentation,
                                                  curGapCount, maxminOW);

                adaptiveTargetAspect = mutatedAsp * 0.5 + adaptiveTargetAspect * 0.5;
                adaptiveTargetAspect = max(0.5, min(2.0, adaptiveTargetAspect));

                int mutW = max(1, (int)round(sqrt((double)totalCells) * sqrt(mutatedAsp)));
                mutW = max(maxminOW, min(ubW, mutW));

                vector<array<int,4>> mutPlace(n);
                int mutH;
                if (packForWidthDynamic(mutW, pieces, mutPlace, mutH, saTime, {}, nullptr, saFragMode)) {
                    long long mutArea = (long long)mutW * mutH;
                    bool isNewBest = mutArea < bestArea ||
                        (mutArea == bestArea && mutH < bestH) ||
                        (mutArea == bestArea && mutH == bestH && mutW < bestW);
                    if (isNewBest) {
                        bestArea = mutArea;
                        bestH = mutH;
                        bestW = mutW;
                        bestPlace = mutPlace;
                        bestWForRefine = mutW;
                        bestAspectRatio = (mutH > 0) ? (double)mutW / mutH : estAspect;
                        stagnation = 0;
                    }
                    updateAdaptiveAspect(mutW, mutH, isNewBest);

                    updateFragModeFromGaps(saFragMode, mutPlace, mutW, mutH);
                }
            }

            int moveType;
            int r = rng() % 100;

            if (inReboot) {
                rebootMode = (rng() % 3);
                const vector<int>* rebootOrdPtr;
                if (rebootMode == 0) rebootOrdPtr = &rebootOrderIrr;
                else if (rebootMode == 1) rebootOrdPtr = &rebootOrderGrid;
                else rebootOrdPtr = &rebootOrderSpace;

                vector<int> shuffledOrd = *rebootOrdPtr;
                if (rng() % 3 == 0) {
                    for (int s = 0; s < n / 3; ++s) {
                        int a = rng() % n, b = rng() % n;
                        swap(shuffledOrd[a], shuffledOrd[b]);
                    }
                }

                int rebootW = curW;
                if (rng() % 2 == 0) {
                    rebootW = aspectBiasedWidthSample(curW, curH, bestAspectRatio);
                } else if (rng() % 3 == 0) {
                    int dW = (int)(rng() % 7) - 3;
                    rebootW = max(maxminOW, min(ubW, curW + dW));
                }
                if (rng() % 4 == 0) {
                    int dW = (int)(rng() % 5) - 2;
                    rebootW = max(maxminOW, min(ubW, estW + dW));
                }
                if (rng() % 5 == 0) {
                    int adW = max(1, (int)round(sqrt((double)totalCells) * sqrt(adaptiveTargetAspect)));
                    int dW = (int)(rng() % 5) - 2;
                    rebootW = max(maxminOW, min(ubW, adW + dW));
                }
                if (rng() % 6 == 0) {
                    int dW = (int)(rng() % 5) - 2;
                    rebootW = max(maxminOW, min(ubW, irrW + dW));
                }
                if (rng() % 7 == 0) {
                    int which = rng() % 4;
                    double aspMetric;
                    if (which == 0) aspMetric = cellWeightedAspect;
                    else if (which == 1) aspMetric = areaWeightedAspect;
                    else if (which == 2) aspMetric = irregWeightedAspect;
                    else aspMetric = compactWeightedAspect;
                    int ppW = max(1, (int)round(sqrt((double)totalCells) * sqrt(aspMetric)));
                    int dW = (int)(rng() % 5) - 2;
                    rebootW = max(maxminOW, min(ubW, ppW + dW));
                }
                if (rng() % 8 == 0) {
                    double ctxAsp = contextualAspectScore(curDensity, curFragmentation);
                    int ctxW = max(1, (int)round(sqrt((double)totalCells) * sqrt(ctxAsp)));
                    int dW = (int)(rng() % 5) - 2;
                    rebootW = max(maxminOW, min(ubW, ctxW + dW));
                }

                double targetAspect = (bestH > 0)
                    ? (double)rebootW / (double)bestH
                    : adaptiveTargetAspect;

                vector<array<int,4>> newPlace(n);
                int newH;
                if (rebootPack(rebootW, pieces, newPlace, newH, rng,
                               saTime, shuffledOrd, rebootMode, targetAspect)) {
                    long long newArea = (long long)rebootW * newH;
                    double newAspect = (newH > 0) ? (double)rebootW / (double)newH : 1.0;
                    bool acceptReboot = false;
                    if (newArea < curArea ||
                        (newArea == curArea && newH < curH)) {
                        acceptReboot = true;
                    } else {
                        double T = T0_narrow * pow(Tmin / T0_narrow, 0.5);
                        double diff = (double)(newArea - curArea) / max(1.0, (double)totalCells);
                        double prob = exp(-diff / max(T, 1e-6));
                        if ((double)(rng() % 10000) / 10000.0 < prob)
                            acceptReboot = true;
                    }

                    if (acceptReboot) {
                        curBias.assign(n, 0);
                        curH = newH;
                        curPlace = newPlace;
                        curArea = newArea;
                        curW = rebootW;
                        curOrder.clear();
                        curFragmentation = computeFragmentation(pieces, curPlace, curW);
                        curDensity = computeDensity(totalCells, curW, curH);
                        curGapCount = computeGapCount(pieces, curPlace, curW, maxminOW, curH);

                        bool isNewBest = newArea < bestArea ||
                            (newArea == bestArea && newH < bestH) ||
                            (newArea == bestArea && newH == bestH && rebootW < bestW);
                        updateAdaptiveAspect(rebootW, newH, isNewBest);

                        if (isNewBest) {
                            bestArea = newArea;
                            bestH = newH;
                            bestW = rebootW;
                            bestPlace = newPlace;
                            bestAspectRatio = newAspect;
                            stagnation = 0;
                        }

                        updateFragModeFromGaps(saFragMode, curPlace, curW, curH);
                    }
                }
                continue;
            }

            if (widePhase) {
                if (r < 35) moveType = 3;
                else if (r < 60) moveType = 0;
                else if (r < 85) moveType = 1;
                else moveType = 4;
            } else {
                if (r < 25) moveType = 3;
                else if (r < 55) moveType = 0;
                else if (r < 80) moveType = 1;
                else moveType = 2;
            }

            if (moveType == 0) {
                int i = pickHighConflictPiece();
                int j = rng() % n;
                int deltaMag = widePhase ? 3 : 1;
                long long delta = (long long)((rng() % (2 * deltaMag + 1)) - deltaMag);
                newBias[i] += delta;
                if (i != j) newBias[j] -= delta;
            } else if (moveType == 1) {
                auto localDynPress = [&](int candW) -> double {
                    return dynPress(candW, curArea, curW, curH, curFragmentation);
                };

                double pMinus = (curW > maxminOW) ? localDynPress(curW - 1) : localDynPress(curW);
                double pPlus = (curW < ubW) ? localDynPress(curW + 1) : localDynPress(curW);
                double gradient = (pPlus - pMinus) / 2.0;

                int gradDir = 0;
                if (gradient < -0.01) gradDir = 1;
                else if (gradient > 0.01) gradDir = -1;

                int numCandidates = 6;
                int maxD = widePhase ? 8 : 3;

                int bestGradW = curW;
                double mostNegativeGrad = 0.0;
                for (int dW = -maxD; dW <= maxD; ++dW) {
                    int candW = curW + dW;
                    if (candW < maxminOW || candW > ubW) continue;
                    if (candW - 1 < maxminOW || candW + 1 > ubW) continue;
                    double g = localDynPress(candW + 1) - localDynPress(candW - 1);
                    if (g < mostNegativeGrad) {
                        mostNegativeGrad = g;
                        bestGradW = candW;
                    }
                }

                double bestCandScore = 1e18;
                int bestCandW = curW;

                for (int c = 0; c < numCandidates; ++c) {
                    int candW;
                    if (c == 0) {
                        candW = bestGradW;
                    } else if (c == 1) {
                        candW = aspectBiasedWidthSample(curW, curH, bestAspectRatio);
                    } else if (c < numCandidates / 2) {
                        int step = gradDir * (1 + (int)(rng() % maxD));
                        step += (int)((rng() % 5) - 2);
                        step = max(-maxD, min(maxD, step));
                        candW = max(maxminOW, min(ubW, curW + step));
                    } else {
                        int s1 = (int)round(gaussDist1(rng));
                        int s2 = (int)round(gaussDist2(rng));
                        int sampled = widePhase ? max(s1, s2) : min(s1, s2);
                        if (widePhase) sampled += (rng() % 3);
                        else sampled -= (rng() % 3);
                        sampled += gradDir * (rng() % 3);
                        int dW = sampled - curW;
                        dW = max(-maxD, min(maxD, dW));
                        candW = max(maxminOW, min(ubW, curW + dW));
                    }

                    double press = localDynPress(candW);
                    double score = (double)candW + 1.5 * press;

                    if (score < bestCandScore) {
                        bestCandScore = score;
                        bestCandW = candW;
                    }
                }

                if (rng() % 7 == 0) {
                    int dW = (int)(rng() % 5) - 2;
                    bestCandW = max(maxminOW, min(ubW, estW + dW));
                }

                if (rng() % 6 == 0) {
                    int adW = max(1, (int)round(sqrt((double)totalCells) * sqrt(adaptiveTargetAspect)));
                    int dW = (int)(rng() % 5) - 2;
                    bestCandW = max(maxminOW, min(ubW, adW + dW));
                }

                if (rng() % 8 == 0) {
                    int dW = (int)(rng() % 5) - 2;
                    bestCandW = max(maxminOW, min(ubW, irrW + dW));
                }

                if (rng() % 9 == 0) {
                    int which = rng() % 4;
                    double aspMetric;
                    if (which == 0) aspMetric = cellWeightedAspect;
                    else if (which == 1) aspMetric = areaWeightedAspect;
                    else if (which == 2) aspMetric = irregWeightedAspect;
                    else aspMetric = compactWeightedAspect;
                    int ppW = max(1, (int)round(sqrt((double)totalCells) * sqrt(aspMetric)));
                    int dW = (int)(rng() % 5) - 2;
                    bestCandW = max(maxminOW, min(ubW, ppW + dW));
                }

                if (rng() % 10 == 0) {
                    double ctxAsp = contextualAspectScore(curDensity, curFragmentation);
                    int ctxW = max(1, (int)round(sqrt((double)totalCells) * sqrt(ctxAsp)));
                    int dW = (int)(rng() % 5) - 2;
                    bestCandW = max(maxminOW, min(ubW, ctxW + dW));
                }

                if (rng() % 5 == 0) {
                    int s1 = (int)round(gaussDist1(rng));
                    int dW = s1 - curW;
                    dW = max(-maxD, min(maxD, dW));
                    bestCandW = max(maxminOW, min(ubW, curW + dW));
                }

                tryW = bestCandW;
            } else if (moveType == 2) {
                if (!curOrder.empty()) {
                    int halfStart = (int)curOrder.size() / 2;
                    int worstPiece = curOrder[halfStart];
                    double worstCS = -1;
                    for (int idx = halfStart; idx < (int)curOrder.size(); ++idx) {
                        int pi = curOrder[idx];
                        if (conflictScore[pi] > worstCS) {
                            worstCS = conflictScore[pi];
                            worstPiece = pi;
                        }
                    }
                    newBias[worstPiece] -= 2;
                    int frontEnd = halfStart;
                    int bestPiece = curOrder[0];
                    double bestCS = 1e18;
                    for (int idx = 0; idx < frontEnd; ++idx) {
                        int pi = curOrder[idx];
                        if (conflictScore[pi] < bestCS) {
                            bestCS = conflictScore[pi];
                            bestPiece = pi;
                        }
                    }
                    newBias[bestPiece] += 1;
                } else {
                    vector<pair<int,int>> pieceY;
                    for (int i = 0; i < n; ++i)
                        pieceY.push_back({curPlace[i][1], i});
                    sort(pieceY.begin(), pieceY.end());
                    if (n >= 2) {
                        newBias[pieceY.back().second] -= 2;
                        newBias[pieceY.front().second] += 1;
                    }
                }
            } else if (moveType == 3) {
                int hi = pickHighConflictPiece();
                int lo = rng() % n;
                for (int tries = 0; tries < 3; ++tries) {
                    int cand = rng() % n;
                    if (conflictScore[cand] < conflictScore[lo]) lo = cand;
                }
                if (hi != lo) {
                    swap(newBias[hi], newBias[lo]);
                } else if (n >= 2) {
                    lo = (hi + 1) % n;
                    swap(newBias[hi], newBias[lo]);
                }
            } else {
                if (!curOrder.empty()) {
                    int halfStart = (int)curOrder.size() / 2;
                    for (int idx = halfStart; idx < (int)curOrder.size(); ++idx) {
                        int pi = curOrder[idx];
                        if (conflictScore[pi] > 0.3) {
                            newBias[pi] -= 1;
                        }
                    }
                } else {
                    int i = pickHighConflictPiece();
                    newBias[i] -= 2;
                }
            }

            vector<array<int,4>> newPlace(n);
            vector<int> newOrder;
            int newH;
            if (packForWidthDynamic(tryW, pieces, newPlace, newH, saTime,
                                    newBias, &newOrder, saFragMode)) {
                long long newArea = (long long)tryW * newH;

                bool accept = false;
                if (newArea < curArea ||
                    (newArea == curArea && newH < curH)) {
                    accept = true;
                } else if (newArea == curArea && newH == curH) {
                    accept = true;
                } else {
                    double T;
                    if (widePhase) {
                        double phase1Frac = frac / phaseBoundary;
                        double phase1Smooth = phase1Frac * phase1Frac * (3.0 - 2.0 * phase1Frac);
                        T = T0_wide * pow(T0_narrow / T0_wide, phase1Smooth);
                    } else {
                        double phase2Frac = (frac - phaseBoundary) / (1.0 - phaseBoundary);
                        double phase2Smooth = phase2Frac * phase2Frac * (3.0 - 2.0 * phase2Frac);
                        T = T0_narrow * pow(Tmin / T0_narrow, phase2Smooth);
                    }
                    double diff = (double)(newArea - curArea) / max(1.0, (double)totalCells);
                    double prob = exp(-diff / max(T, 1e-6));
                    if ((double)(rng() % 10000) / 10000.0 < prob)
                        accept = true;
                }

                if (accept) {
                    curBias = newBias;
                    curH = newH;
                    curPlace = newPlace;
                    curArea = newArea;
                    curW = tryW;
                    curOrder = newOrder;
                    curFragmentation = computeFragmentation(pieces, curPlace, curW);
                    curDensity = computeDensity(totalCells, curW, curH);
                    curGapCount = computeGapCount(pieces, curPlace, curW, maxminOW, curH);

                    updateFragModeFromGaps(saFragMode, curPlace, curW, curH);

                    bool isNewBest = newArea < bestArea ||
                        (newArea == bestArea && newH < bestH) ||
                        (newArea == bestArea && newH == bestH && tryW < bestW);
                    updateAdaptiveAspect(tryW, newH, isNewBest);

                    if (isNewBest) {
                        bestArea = newArea;
                        bestH = newH;
                        bestW = tryW;
                        bestPlace = newPlace;
                        bestAspectRatio = (newH > 0) ? (double)tryW / newH : estAspect;
                        stagnation = 0;
                    } else {
                        stagnation++;
                    }

                    if (widePhase) {
                        if (newArea < phase1BestArea ||
                            (newArea == phase1BestArea && newH < phase1BestH)) {
                            phase1BestArea = newArea;
                            phase1BestH = newH;
                            phase1BestW = tryW;
                            phase1BestPlace = newPlace;
                        }
                    }
                } else {
                    stagnation++;
                }

                if (widePhase && frac >= phaseBoundary && !phaseTransitioned) {
                    phaseTransitioned = true;
                    if (phase1BestArea <= curArea) {
                        curBias.assign(n, 0);
                        curH = phase1BestH;
                        curPlace = phase1BestPlace;
                        curArea = phase1BestArea;
                        curW = phase1BestW;
                        curOrder.clear();
                        curFragmentation = computeFragmentation(pieces, curPlace, curW);
                        curDensity = computeDensity(totalCells, curW, curH);
                        curGapCount = computeGapCount(pieces, curPlace, curW, maxminOW, curH);
                        updateFragModeFromGaps(saFragMode, curPlace, curW, curH);
                    }
                    stagnation = 0;

                    double pMinusT = (curW > maxminOW)
                        ? dynPress(curW - 1, curArea, curW, curH, curFragmentation)
                        : dynPress(curW, curArea, curW, curH, curFragmentation);
                    double pPlusT = (curW < ubW)
                        ? dynPress(curW + 1, curArea, curW, curH, curFragmentation)
                        : dynPress(curW, curArea, curW, curH, curFragmentation);
                    double gradMag = fabs(pPlusT - pMinusT) / 2.0;

                    double rebootDuration;
                    if (gradMag < 0.01) {
                        rebootDuration = 0.15;
                    } else {
                        rebootDuration = max(0.03, 0.15 / (1.0 + gradMag * 10.0));
                    }
                    rebootDuration = min(rebootDuration, saTime - elapsed() - 0.05);
                    if (rebootDuration > 0.01) {
                        inReboot = true;
                        rebootEndTime = elapsed() + rebootDuration;
                    }
                }

                if (stagnation > 50 && !inReboot) {
                    for (int i = 0; i < n; ++i)
                        curBias[i] = (long long)((rng() % 9) - 4);
                    stagnation = 0;

                    double gradMagStag = fabs(
                        dynPress(curW + 1, curArea, curW, curH, curFragmentation) -
                        dynPress(max(maxminOW, curW - 1), curArea, curW, curH, curFragmentation)
                    ) / 2.0;
                    double rebootDur = (gradMagStag < 0.01) ? 0.08
                        : max(0.02, 0.08 / (1.0 + gradMagStag * 10.0));
                    rebootDur = min(rebootDur, saTime - elapsed() - 0.05);
                    if (rebootDur > 0.01) {
                        inReboot = true;
                        rebootEndTime = elapsed() + rebootDur;
                    }
                }
            }
        }
    }

    cout << bestW << ' ' << bestH << '\n';
    for (const auto& p : bestPlace) {
        cout << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << p[3] << '\n';
    }
    return 0;
}