#define MEASURE_FUNCTION_TIME
#include <pcl/visualization/range_image_visualizer.h>
#include "pcl/io/openni2/openni.h"
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/convex_hull.h>
#include <boost/chrono.hpp>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/time.h> //fps calculations
#include <pcl/common/angles.h>
#include <pcl/io/openni2_grabber.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/boost.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <iostream>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;
using namespace pcl;
typedef boost::chrono::high_resolution_clock HRClock;

#define SHOW_FPS 1
#if SHOW_FPS
#define FPS_CALC(_WHAT_) \
  do \
{ \
  static unsigned count = 0;\
  static double last = pcl::getTime ();\
  double now = pcl::getTime (); \
  ++count; \
  if (now - last >= 1.0) \
{ \
  std::cout << "Average framerate ("<< _WHAT_ << "): " << double (count)/double (now - last) << " Hz" <<  std::endl; \
  count = 0; \
  last = now; \
} \
}while (false)
#else
#define FPS_CALC (_WHAT_) \
  do \
{ \
}while (false)
#endif

void
printHelp (int, char **argv)
{
  using pcl::console::print_error;
  using pcl::console::print_info;

  print_error ("Syntax is: %s [((<device_id> | <path-to-oni-file>) [-depthmode <mode>] [-imagemode <mode>] [-xyz] | -l [<device_id>]| -h | --help)]\n", argv [0]);
  print_info ("%s -h | --help : shows this help\n", argv [0]);
  print_info ("%s -xyz : use only XYZ values and ignore RGB components (this flag is required for use with ASUS Xtion Pro) \n", argv [0]);
  print_info ("%s -l : list all available devices\n", argv [0]);
  print_info ("%s -l <device-id> :list all available modes for specified device\n", argv [0]);
  print_info ("\t\t<device_id> may be \"#1\", \"#2\", ... for the first, second etc device in the list\n");
#ifndef _WIN32
  print_info ("\t\t                   bus@address for the device connected to a specific usb-bus / address combination\n");
  print_info ("\t\t                   <serial-number>\n");
#endif
  print_info ("\n\nexamples:\n");
  print_info ("%s \"#1\"\n", argv [0]);
  print_info ("\t\t uses the first device.\n");
  print_info ("%s  \"./temp/test.oni\"\n", argv [0]);
  print_info ("\t\t uses the oni-player device to play back oni file given by path.\n");
  print_info ("%s -l\n", argv [0]);
  print_info ("\t\t list all available devices.\n");
  print_info ("%s -l \"#2\"\n", argv [0]);
  print_info ("\t\t list all available modes for the second device.\n");
#ifndef _WIN32
  print_info ("%s A00361800903049A\n", argv [0]);
  print_info ("\t\t uses the device with the serial number \'A00361800903049A\'.\n");
  print_info ("%s 1@16\n", argv [0]);
  print_info ("\t\t uses the device on address 16 at USB bus 1.\n");
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointType>
class OpenNI2Viewer
{
public:
  typedef pcl::PointCloud<PointType> Cloud;
  typedef typename Cloud::ConstPtr CloudConstPtr;
  typedef typename Cloud::Ptr CloudPtr;
  typedef pcl::search::KdTree<PointType> TreeType;
  typedef typename TreeType::Ptr TreeTypePtr;

  OpenNI2Viewer (pcl::io::OpenNI2Grabber& grabber)
    : cloud_viewer_ (new pcl::visualization::PCLVisualizer ("PCL OpenNI2 cloud"))
    , image_viewer_ ()
    , grabber_ (grabber)
    , rgb_data_ (0), rgb_data_size_ (0)
  {
  }

  void
  cloud_callback (const CloudConstPtr& cloud)
  {
    FPS_CALC ("cloud callback");
    boost::mutex::scoped_lock lock (cloud_mutex_);
    cloud_ = cloud;
  }

  void
  image_callback (const boost::shared_ptr<pcl::io::openni2::Image>& image)
  {
    FPS_CALC ("image callback");
    boost::mutex::scoped_lock lock (image_mutex_);
    image_ = image;

    if (image->getEncoding () != pcl::io::openni2::Image::RGB)
    {
      if (rgb_data_size_ < image->getWidth () * image->getHeight ())
      {
        if (rgb_data_)
          delete [] rgb_data_;
        rgb_data_size_ = image->getWidth () * image->getHeight ();
        rgb_data_ = new unsigned char [rgb_data_size_ * 3];
      }
      image_->fillRGB (image_->getWidth (), image_->getHeight (), rgb_data_);
    }
  }

  void
  keyboard_callback (const pcl::visualization::KeyboardEvent& event, void*)
  {
    if (event.getKeyCode ())
      cout << "the key \'" << event.getKeyCode () << "\' (" << event.getKeyCode () << ") was";
    else
      cout << "the special key \'" << event.getKeySym () << "\' was";
    if (event.keyDown ())
      cout << " pressed" << endl;
    else
      cout << " released" << endl;
  }

  void
  mouse_callback (const pcl::visualization::MouseEvent& mouse_event, void*)
  {
    if (mouse_event.getType () == pcl::visualization::MouseEvent::MouseButtonPress && mouse_event.getButton () == pcl::visualization::MouseEvent::LeftButton)
    {
      cout << "left button pressed @ " << mouse_event.getX () << " , " << mouse_event.getY () << endl;
    }
  }

  /**
  * @brief starts the main loop
  */
  void
  run ()
  {
    cloud_viewer_->registerMouseCallback (&OpenNI2Viewer::mouse_callback, *this);
    cloud_viewer_->registerKeyboardCallback (&OpenNI2Viewer::keyboard_callback, *this);
    cloud_viewer_->setCameraFieldOfView (1.02259994f);
    boost::function<void (const CloudConstPtr&) > cloud_cb = boost::bind (&OpenNI2Viewer::cloud_callback, this, _1);
    boost::signals2::connection cloud_connection = grabber_.registerCallback (cloud_cb);

    boost::signals2::connection image_connection;
    if (grabber_.providesCallback<void (const boost::shared_ptr<pcl::io::openni2::Image>&)>())
    {
      image_viewer_.reset (new pcl::visualization::ImageViewer ("PCL OpenNI image"));
      image_viewer_->registerMouseCallback (&OpenNI2Viewer::mouse_callback, *this);
      image_viewer_->registerKeyboardCallback (&OpenNI2Viewer::keyboard_callback, *this);
      boost::function<void (const boost::shared_ptr<pcl::io::openni2::Image>&) > image_cb = boost::bind (&OpenNI2Viewer::image_callback, this, _1);
      image_connection = grabber_.registerCallback (image_cb);
    }

    bool image_init = false, cloud_init = false;

    grabber_.start ();

    while (!cloud_viewer_->wasStopped () && (image_viewer_ && !image_viewer_->wasStopped ()))
    {
      boost::shared_ptr<pcl::io::openni2::Image> image;
      CloudConstPtr cloud;
      cloud_viewer_->spinOnce ();
      if (cloud_mutex_.try_lock ())
      {
        cloud_.swap (cloud);
        cloud_mutex_.unlock ();
      }

      if (cloud)
      {
          FPS_CALC("drawing cloud");
          if (!cloud_init)
          {
              cloud_viewer_->setPosition (0, 0);
              cloud_viewer_->setSize (cloud->width, cloud->height);
              cloud_init = !cloud_init;
          }
        if (!cloud_viewer_->updatePointCloud (cloud, "OpenNICloud"))
        {
            cloud_viewer_->addPointCloud (cloud, "OpenNICloud");
            cloud_viewer_->resetCameraViewpoint ("OpenNICloud");
            cloud_viewer_->setCameraPosition (
                    0,0,0,
                    0,0,1,
                    0,-1,0);
        }
            CloudPtr filtered(new Cloud);
            CloudPtr cloud_rest(new Cloud);
            CloudPtr cloud_plane(new Cloud);
      	    CloudPtr objects (new Cloud);
            pcl::VoxelGrid<PointType> p;
            p.setInputCloud (cloud);
            p.setLeafSize (0.01f, 0.01f, 0.01f);
            p.filter(*filtered);
            pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
            pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
            pcl::SACSegmentation<PointType> seg;
            seg.setOptimizeCoefficients (true);
            seg.setModelType (pcl::SACMODEL_PLANE);
            seg.setMethodType (pcl::SAC_RANSAC);
            seg.setMaxIterations (1000);
            seg.setDistanceThreshold (0.01);
////////////////////////
            pcl::ExtractIndices<PointType> extract;
            seg.setInputCloud (filtered);
            seg.segment (*inliers, *coefficients);
            if (inliers->indices.size () == 0)
            {
                std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
                break;
            }
            extract.setInputCloud (filtered);
            extract.setIndices (inliers);
            extract.setNegative (false);
            extract.filter (*cloud_plane);
//
      	    PointIndices::Ptr indices_but_the_plane (new PointIndices);
            extract.getRemovedIndices (*indices_but_the_plane);
      	    PointIndices::Ptr points_above_plane (new PointIndices);
//
    	    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

            pcl::ProjectInliers<PointType> proj;
            proj.setModelType (pcl::SACMODEL_PLANE);
            proj.setIndices (inliers);
            proj.setInputCloud (filtered);
            proj.setModelCoefficients (coefficients);
            proj.filter (*cloud_plane);
            pcl::visualization::PointCloudColorHandlerCustom<PointType> green(cloud_plane, 0, 255, 0);
    	    std::cout << "PointCloud representing after projection component: " << cloud_plane->points.size () << " data points." << std::endl;
            if (!cloud_viewer_->updatePointCloud (cloud_plane, green, "Planar"))
            {
                cloud_viewer_->addPointCloud (cloud_plane, green, "Planar");
            }
            CloudPtr cloud_hull (new Cloud);
            pcl::ConvexHull<PointType> chull;
            chull.setInputCloud (cloud_plane);
            chull.reconstruct (*cloud_hull);
            pcl::visualization::PointCloudColorHandlerCustom<PointType> blue(cloud_hull, 0, 255, 0);
            if (!cloud_viewer_->updatePointCloud (cloud_hull, blue, "Hull"))
            {
                cloud_viewer_->addPointCloud (cloud_hull, blue, "Hull");
            }
            pcl::ExtractPolygonalPrismData<PointType> eppd;
            eppd.setInputCloud (cloud);
            eppd.setIndices (indices_but_the_plane);
            eppd.setInputPlanarHull (cloud_hull);
            eppd.setHeightLimits (0.001, 0.5);
            eppd.segment(*points_above_plane);
            extract.setIndices(points_above_plane);
            extract.filter(*objects);


    	    std::cout << "PointCloud representing after projection component: " << objects->points.size () << " data points." << std::endl;
 	    pcl::visualization::PointCloudColorHandlerCustom<PointType> red(cloud_rest, 255, 0, 0);
            if (!cloud_viewer_->updatePointCloud (objects,red, "objects"))
            {
                cloud_viewer_->addPointCloud (objects,red, "objects");
            }


            extract.setNegative (true);
            extract.filter (*cloud_rest);
            filtered.swap (cloud_rest);
            pcl::visualization::PointCloudColorHandlerCustom<PointType> yellow(cloud_rest, 0, 255, 0);
            if (!cloud_viewer_->updatePointCloud (cloud_rest,yellow, "rest"))
            {
                cloud_viewer_->addPointCloud (cloud_rest,yellow, "rest");
            }
            TreeTypePtr tree (new TreeType);
            tree->setInputCloud (cloud_rest);
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<PointType> ec;
            ec.setClusterTolerance (0.02);
            ec.setMinClusterSize (100);
            ec.setMaxClusterSize (25000);
            ec.setSearchMethod (tree);
            ec.setInputCloud (cloud_rest);
            ec.extract (cluster_indices);
            int i = 0;
            pcl::PCDWriter writer;
            for(int count=0;count<100;count++)
            {
                double now = pcl::getTime ();
                char *dirName=new char[20] ;
                int status;
                cout<<"the "<<count<<" is saved"<<endl;
                sprintf(dirName,"%f",now,count);
                status=mkdir(dirName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
                {
                    CloudPtr cloud_cluster (new Cloud);
                    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
                        cloud_cluster->points.push_back (cloud_rest->points[*pit]);
                    cloud_cluster->width = cloud_cluster->points.size ();
                    cloud_cluster->height = 1;
                    cloud_cluster->is_dense = true;
                    pcl::visualization::PointCloudColorHandlerRandom<PointType> random(cloud_cluster);
                    char filename[20] = {0};
                    char filename_temp[20] = {0};
                    static int i=0;
                    std::cout<<"the "<<i<<" is saved"<<std::endl;
                    std::cout<<"dirName= "<<dirName<<std::endl;
                    sprintf(filename_temp,"./%s/",dirName);
                    sprintf(filename,"%s%d.pcd",filename_temp,i);
                    std::cout<<"filename= "<<filename<<std::endl;
                    if (!cloud_viewer_->updatePointCloud (cloud_cluster, random, filename))
                    {
                        cloud_viewer_->addPointCloud (cloud_cluster, random, filename);
                    }
                    pcl::io::savePCDFile (filename, *cloud_cluster);
                    i++;
                }//cluster循环结束
            }//for文件夹循环结束
      }//ifcloud循环结束
      if (image_mutex_.try_lock ())
      {
          image_.swap (image);
          image_mutex_.unlock ();
      }
      if (image)
      {
          if (!image_init && cloud && cloud->width != 0)
          {
              image_viewer_->setPosition (cloud->width, 0);
              image_viewer_->setSize (cloud->width, cloud->height);
              image_init = !image_init;
          }
          if (image->getEncoding () == pcl::io::openni2::Image::RGB)
              image_viewer_->addRGBImage ( (const unsigned char*)image->getData (), image->getWidth (), image->getHeight ());
          else
              image_viewer_->addRGBImage (rgb_data_, image->getWidth (), image->getHeight ());
          image_viewer_->spinOnce ();
      }
    }//while循环结束
    grabber_.stop ();
    cloud_connection.disconnect ();
    image_connection.disconnect ();
    if (rgb_data_)
        delete[] rgb_data_;
  }//voidrun结束
  boost::shared_ptr<pcl::visualization::PCLVisualizer> cloud_viewer_;
  boost::shared_ptr<pcl::visualization::ImageViewer> image_viewer_;
  pcl::io::OpenNI2Grabber& grabber_;
  boost::mutex cloud_mutex_;
  boost::mutex image_mutex_;
  CloudConstPtr cloud_;
  boost::shared_ptr<pcl::io::openni2::Image> image_;
  unsigned char* rgb_data_;
  unsigned rgb_data_size_;
};
boost::shared_ptr<pcl::visualization::PCLVisualizer> cld;
boost::shared_ptr<pcl::visualization::ImageViewer> img;
int main (int argc, char** argv)
{
  std::string device_id ("");
  pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
  pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
  bool xyz = false;

  if (argc >= 2)
  {
    device_id = argv[1];
    if (device_id == "--help" || device_id == "-h")
    {
      printHelp (argc, argv);
      return 0;
    }
    else if (device_id == "-l")
    {
      if (argc >= 3)
      {
        pcl::io::OpenNI2Grabber grabber (argv[2]);
        boost::shared_ptr<pcl::io::openni2::OpenNI2Device> device = grabber.getDevice ();
        cout << *device;		// Prints out all sensor data, including supported video modes
      }
      else
      {
        boost::shared_ptr<pcl::io::openni2::OpenNI2DeviceManager> deviceManager = pcl::io::openni2::OpenNI2DeviceManager::getInstance ();
        if (deviceManager->getNumOfConnectedDevices () > 0)
        {
          for (unsigned deviceIdx = 0; deviceIdx < deviceManager->getNumOfConnectedDevices (); ++deviceIdx)
          {
            boost::shared_ptr<pcl::io::openni2::OpenNI2Device> device = deviceManager->getDeviceByIndex (deviceIdx);
            cout << "Device " << device->getStringID () << "connected." << endl;
          }

        }
        else
          cout << "No devices connected." << endl;

        cout <<"Virtual Devices available: ONI player" << endl;
      }
      return 0;
    }
  }
  else
  {
    boost::shared_ptr<pcl::io::openni2::OpenNI2DeviceManager> deviceManager = pcl::io::openni2::OpenNI2DeviceManager::getInstance ();
    if (deviceManager->getNumOfConnectedDevices () > 0)
    {
      boost::shared_ptr<pcl::io::openni2::OpenNI2Device> device = deviceManager->getAnyDevice ();
      cout << "Device ID not set, using default device: " << device->getStringID () << endl;
    }
  }

  unsigned mode;
  if (pcl::console::parse (argc, argv, "-depthmode", mode) != -1)
    depth_mode = pcl::io::OpenNI2Grabber::Mode (mode);

  if (pcl::console::parse (argc, argv, "-imagemode", mode) != -1)
    image_mode = pcl::io::OpenNI2Grabber::Mode (mode);

  if (pcl::console::find_argument (argc, argv, "-xyz") != -1)
    xyz = true;

  try
  {
    pcl::io::OpenNI2Grabber grabber (device_id, depth_mode, image_mode);

    if (xyz || !grabber.providesCallback<pcl::io::OpenNI2Grabber::sig_cb_openni_point_cloud_rgb> ())
    {
      OpenNI2Viewer<pcl::PointXYZ> openni_viewer (grabber);
      openni_viewer.run ();
    }
    else
    {
      OpenNI2Viewer<pcl::PointXYZRGBA> openni_viewer (grabber);
      openni_viewer.run ();
    }
  }
  catch (pcl::IOException& e)
  {
    pcl::console::print_error ("Failed to create a grabber: %s\n", e.what ());
    return (1);
  }

  return (0);
}
/* ]--- */
