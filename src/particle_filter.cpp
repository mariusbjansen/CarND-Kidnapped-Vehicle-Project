/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // DONE: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).

  normal_distribution<double> distrib_x{0, std[0]};
  normal_distribution<double> distrib_y{0, std[1]};
  normal_distribution<double> distrib_theta{0, std[2]};

  num_particles = 100;
  particles.resize(num_particles);

  for (auto &particle : particles) {
    particle.x = x + distrib_x(gen);
    particle.y = y + distrib_y(gen);
    particle.theta = theta + distrib_theta(gen);
    particle.weight = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // DONE: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  for (auto &particle : particles) {
    // deterministic prediction with numerical stability check ()

    if (fabs(yaw_rate) < 1E-4) {
      // equations from the Udacity fusion model
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
      particle.theta += yaw_rate * delta_t;
    } else {
      particle.x +=
          velocity / yaw_rate *
          (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));

      particle.y +=
          velocity / yaw_rate *
          (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));

      particle.theta += yaw_rate * delta_t;
    }
    // adding noise
    std::normal_distribution<double> distrib_x{particle.x, std_pos[0]};
    std::normal_distribution<double> distrib_y{particle.y, std_pos[1]};
    std::normal_distribution<double> distrib_theta{particle.theta, std_pos[2]};

    particle.x = distrib_x(gen);
    particle.y = distrib_y(gen);
    particle.theta = distrib_theta(gen);
  }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs> landmarks,
                                     std::vector<LandmarkObs> &observations) {
  // DONE: Find the landmarks measurement that is closest to each observed
  // measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to
  //   implement this method and use it as a helper during the updateWeights
  //   phase.

  for (auto &obs : observations) {
    double dist_min = numeric_limits<double>::max();

    for (auto pred : landmarks) {
      double dist_curr = dist(obs.x, obs.y, pred.x, pred.y);

      if (dist_curr < dist_min) {
        obs.id = pred.id;
        dist_min = dist_curr;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  // DONE: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems.
  //   Keep in mind that this transformation requires both rotation AND
  //   translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  // transform map to observation list

  // for each particle transform observations to gloabl coordinate system
  for (auto &particle : particles) {
    vector<LandmarkObs> landmarks;
    for (auto landmark : map_landmarks.landmark_list) {
      LandmarkObs lm;
      lm.x = landmark.x_f;
      lm.y = landmark.y_f;
      lm.id = landmark.id_i;

      if ((fabs(lm.x - particle.x) <= sensor_range) &&
          (fabs(lm.y - particle.y) <= sensor_range)) {
        landmarks.push_back(lm);
      }
    }

    std::vector<LandmarkObs> particles_obs_global;
    for (auto obs : observations) {
      LandmarkObs obs_global;
      obs_global.x = cos(particle.theta) * obs.x - sin(particle.theta) * obs.y +
                     particle.x;
      obs_global.y = sin(particle.theta) * obs.x + cos(particle.theta) * obs.y +
                     particle.y;

      particles_obs_global.push_back(obs_global);
    }

    // data association with result that particles_obs_global has most likely ID
    dataAssociation(landmarks, particles_obs_global);

    // weighting
    particle.weight = 1.0;

    for (auto particle_obs_glob : particles_obs_global) {
      for (auto landmark : landmarks) {
        if (particle_obs_glob.id == landmark.id) {
          double po_x, po_y, lm_x, lm_y, std_x, std_y, fact;
          po_x = particle_obs_glob.x;
          po_y = particle_obs_glob.y;
          lm_x = landmark.x;
          lm_y = landmark.y;
          std_x = std_landmark[0];
          std_y = std_landmark[1];

          fact = 1 / (2 * M_PI * std_x * std_y) *
                 exp(-0.5 * (pow((lm_x - po_x) / std_x, 2) +
                             pow((lm_y - po_y) / std_y, 2)));

          particle.weight *= fact;
        }
      }
    }
  }
}

void ParticleFilter::resample() {
  // DONE: Resample particles with replacement with probability proportional to
  // their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // currently global weights are empty -> copy to member
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // starting index using uniform random distribution int
  uniform_int_distribution<> uni_distribution_int(0, num_particles - 1);
  auto index = uni_distribution_int(gen);

  // distribution for drawing using uniform random distribution double
  uniform_real_distribution<> uni_distribution_double(0.0, 1.0);

  auto mw = *max_element(weights.begin(), weights.end());
  double beta = 0.0;

  vector<Particle> resampled;

  for (auto particle : particles) {
    beta += uni_distribution_double(gen) * 2.0 * mw;
    while (beta > particles.at(index).weight) {
      beta -= particles.at(index).weight;
      index = (index + 1) % num_particles;
    }
    resampled.push_back(particles.at(index));
  }

  particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle &particle,
                                         const std::vector<int> &associations,
                                         const std::vector<double> &sense_x,
                                         const std::vector<double> &sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
