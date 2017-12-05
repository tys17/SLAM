#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
  T* projectPoint(T R11, T R12, T R13, T R21, T R22, T R23, T R31, T R32, T R33, 
                  T t1, T t2, T t3, 
                  T point_x, T point_y, T point_z){
      T p[3];
      p[0] = (R11*point_x + R12*point_y + R13*point_z) + t1;
      p[1] = (R21*point_x + R22*point_y + R23*point_z) + t2;
      p[2] = (R31*point_x + R32*point_y + R33*point_z) + t3;

      p[0] = K[0]*p[0]+K[1]*p[1]+K[2]*p[2];
      p[1] = K[3]*p[0]+K[4]*p[1]+K[5]*p[2];
      p[2] = K[6]*p[0]+K[7]*p[1]+K[8]*p[2];

      p[0] = p[0]/p[2];
      p[1] = p[1]/p[2];

      T pi[2];
      pi[0] = p[0];
      pi[1] = p[1];
      return pi;
  }

  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T const R11, const T const R12, const T const R13,
                  const T const R21, const T const R22, const T const R23,
                  const T const R31, const T const R32, const T const R33,
                  const T const t1, const T const t2, const T const t3,
                  const T const point_x, const T const point_y, const T const point_z,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[2];
    p = projectPoint(R11,R12,R13,R21,R22,R23,R31,R32,R33, t1,t2,t3, point_x, point_y, point_z);


    // The error is the difference between the predicted and observed position.
    residuals[0] = p[0] - observed_x;
    residuals[1] = p[1] - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2,
                1,1,1,1,1,1,1,1,1,  1,1,1,  1,1,1>(
                new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

int BA(int start_frame, std::vector<frame *> & kfDB, std::vector<feature *> & ftDB) {

  ceres::Problem problem;
  for (int cur_frame = start_frame; cur_frame<R.size(); cur_frame++){

    int cur_frame_id = kfDB[cur_frame].frame_id;

    for (int cur_feat = 0; cur_feat < kfDB[cur_frame].feature_id_vec.size(); ++cur_feat) {

      int cur_feat_id = kfDB[cur_frame].feature_id_vec[cur_feat].feature_id;
      int cur_feat_x = ftDB[cur_feat_id].xycoord[cur_frame_id].x;
      int cur_feat_y = ftDB[cur_feat_id].xycoord[cur_frame_id].y;


      ceres::CostFunction* cost_function =
          SnavelyReprojectionError::Create(cur_feat_x,
                                           cur_feat_y);
      problem.AddResidualBlock(cost_function,
                               NULL /* squared loss */,
                               &kfDB[cur_frame].R.at(0,0), 
                               &kfDB[cur_frame].R.at(0,1),
                               &kfDB[cur_frame].R.at(0,2),
                               &kfDB[cur_frame].R.at(1,0), 
                               &kfDB[cur_frame].R.at(1,1),
                               &kfDB[cur_frame].R.at(1,2),
                               &kfDB[cur_frame].R.at(2,0), 
                               &kfDB[cur_frame].R.at(2,1),
                               &kfDB[cur_frame].R.at(2,2),
                               &kfDB[cur_frame].t.at(0,0), 
                               &kfDB[cur_frame].t.at(1,0),
                               &kfDB[cur_frame].t.at(2,0),
                               &ftDB[cur_feat_id].coord3D.x,
                               &ftDB[cur_feat_id].coord3D.y,
                               &ftDB[cur_feat_id].coord3D.z);
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  return 0;
}
