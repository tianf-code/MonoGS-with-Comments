import open3d as o3d

pcd = o3d.io.read_point_cloud("results/Datasets_tum_short/2024-03-26-20-40-00/point_cloud/final/point_cloud.ply")
# vis = o3d.visualization.Visualizer()
# vis.create_window()

pcd.paint_uniform_color(color=[1, 1, 0])
o3d.visualization.draw_geometries([pcd])

# vis.poll_events()
# vis.update_renderer()
# vis.run()