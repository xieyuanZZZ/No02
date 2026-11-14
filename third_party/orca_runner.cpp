// ORCA/RVO2 runner with acceleration-limited velocity tracking.
// Official RVO2 core is used for the ORCA velocity decision; the runner does
// NOT teleport agents: desired velocity is tracked with |a|<=a_max, and the
// resulting actual state is fed back into RVO2 every step.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "RVO.h"

struct Vec { double x, y; };
static double norm(Vec a) { return std::hypot(a.x, a.y); }

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: orca_runner scenario.txt\n"; return 2; }
    std::ifstream f(argv[1]);
    int n, steps; double dt, vmax, amax, radius, neighbor, horizon;
    f >> n >> dt >> steps >> vmax >> amax >> radius >> neighbor >> horizon;
    std::vector<std::vector<Vec>> paths(n);
    std::vector<Vec> pos(n), vel(n, {0, 0});
    for (int i = 0; i < n; ++i) {
        int m; f >> m; paths[i].resize(m);
        for (int k = 0; k < m; ++k) f >> paths[i][k].x >> paths[i][k].y;
        pos[i] = paths[i][0];
    }
    int ne; f >> ne;
    struct Ev { double t; int id; double dur; };
    std::vector<Ev> evs(ne);
    for (int k = 0; k < ne; ++k) f >> evs[k].t >> evs[k].id >> evs[k].dur;
    int nobs; f >> nobs;
    std::vector<std::vector<RVO::Vector2>> obstacles(nobs);
    for (int o = 0; o < nobs; ++o) {
        int nv; f >> nv;
        obstacles[o].resize(nv);
        for (int k = 0; k < nv; ++k) {
            double x, y; f >> x >> y;
            obstacles[o][k] = RVO::Vector2((float)x, (float)y);
        }
    }

    RVO::RVOSimulator sim;
    sim.setTimeStep((float)dt);
    sim.setAgentDefaults((float)neighbor, 10, (float)horizon, (float)horizon, (float)radius, (float)vmax);
    for (const auto& poly : obstacles) sim.addObstacle(poly);
    if (!obstacles.empty()) sim.processObstacles();
    std::vector<size_t> ids;
    std::vector<int> target(n, 1);
    std::vector<double> pause_until(n, -1.0);
    for (int i = 0; i < n; ++i) {
        ids.push_back(sim.addAgent(RVO::Vector2((float)pos[i].x, (float)pos[i].y)));
        sim.setAgentMaxSpeed(ids.back(), (float)vmax);
        sim.setAgentRadius(ids.back(), (float)radius);
        sim.setAgentNeighborDist(ids.back(), (float)neighbor);
        sim.setAgentTimeHorizon(ids.back(), (float)horizon);
    }
    std::cout << "step,robot,x,y,vx,vy,target,paused\n";
    for (int s = 0; s <= steps; ++s) {
        double t = s * dt;
        for (const auto& e : evs) if (std::abs(e.t - t) < 1e-9 && e.id >= 0 && e.id < n)
            pause_until[e.id] = t + e.dur;
        for (int i = 0; i < n; ++i) {
            bool paused = t < pause_until[i] - 1e-9;
            if (!paused) {
                while (target[i] < (int)paths[i].size()) {
                    Vec d{paths[i][target[i]].x - pos[i].x,
                          paths[i][target[i]].y - pos[i].y};
                    if (norm(d) <= 0.12) ++target[i];
                    else break;
                }
            }
            Vec pref{0, 0};
            if (!paused && target[i] < (int)paths[i].size()) {
                Vec d{paths[i][target[i]].x - pos[i].x,
                      paths[i][target[i]].y - pos[i].y};
                double L = std::max(norm(d), 1e-9);
                double sp = std::min(vmax, L / dt);
                pref = {d.x / L * sp, d.y / L * sp};
            } else if (!paused && target[i] >= (int)paths[i].size()) {
                // Hold the actual endpoint; the target's 0.12 m acceptance
                // radius otherwise leaves the robot 0.3-0.4 m short.
                Vec d{paths[i].back().x - pos[i].x,
                      paths[i].back().y - pos[i].y};
                double L = norm(d);
                if (L > 0.03) {
                    double sp = std::min(vmax, L / dt);
                    pref = {d.x / L * sp, d.y / L * sp};
                }
            }
            sim.setAgentPrefVelocity(ids[i], RVO::Vector2((float)pref.x, (float)pref.y));
        }
        sim.doStep();
        for (int i = 0; i < n; ++i) {
            RVO::Vector2 vd = sim.getAgentVelocity(ids[i]);
            Vec dv{vd.x() - vel[i].x, vd.y() - vel[i].y};
            double L = norm(dv);
            double maxdv = amax * dt;
            if (L > maxdv && L > 1e-12) { dv.x *= maxdv / L; dv.y *= maxdv / L; }
            vel[i].x += dv.x; vel[i].y += dv.y;
            double sp = norm(vel[i]);
            if (sp > vmax) { vel[i].x *= vmax / sp; vel[i].y *= vmax / sp; }
            pos[i].x += vel[i].x * dt;
            pos[i].y += vel[i].y * dt;
            sim.setAgentPosition(ids[i], RVO::Vector2((float)pos[i].x, (float)pos[i].y));
            sim.setAgentVelocity(ids[i], RVO::Vector2((float)vel[i].x, (float)vel[i].y));
            bool paused = t < pause_until[i] - 1e-9;
            std::printf("%d,%d,%.9f,%.9f,%.9f,%.9f,%d,%d\n", s, i, pos[i].x,
                        pos[i].y, vel[i].x, vel[i].y, target[i], paused ? 1 : 0);
        }
    }
    return 0;
}
