#######
## Convert CSV file containing lines coordinates 
## in the format (for each row): line_id, coord_x, coord_y 
## to VTP file format (which can be opened with Paraview)
##
## @author Steve Petruzza
######


import vtk
import csv
import sys

# write here the name of the features (colums after id, x, y, z)
features_name = ['x','y']
n_coords = 2

#########################################################

if len(sys.argv) < 2:
    print("Invalid parameters:")
    print("Usage: $", sys.argv[0], "<input-csv-file>")
    exit(0)

input_csv = sys.argv[1]

# points in all polylines
points = vtk.vtkPoints()

fids = [] # fiber ids for each point

fibers_len = dict() # len of each fiber

n_points=0

last_id = -1

with open(input_csv) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV) #skip header

    for row in readCSV:
      p = [float(row[1]), float(row[2]), 0.0]  # reading the position of the point in the line
      
      points.InsertNextPoint(p)

      fids.append(row[0])
      
      if not fids[n_points] in fibers_len:
        fibers_len[fids[n_points]] = 1
      else:
        fibers_len[fids[n_points]] += 1

      n_points += 1 

print(fibers_len)

polylines = dict()

for k, v in fibers_len.items():
  polylines[k] = vtk.vtkPolyLine()
  polylines[k].GetPointIds().SetNumberOfIds(v)

last_id = -1
curr_id = 0

for i in range(0, n_points):
  if(fids[i] != last_id):
    last_id = fids[i]
    curr_id = 0

  #print(i,curr_id, polylines[fids[i]].GetPointIds().GetNumberOfIds())
  polylines[fids[i]].GetPointIds().SetId(curr_id, i) # set id of the point inside the line

  curr_id += 1

# Create a cell array to store the lines in and add the lines to it
cells = vtk.vtkCellArray()

for p in polylines.values():
  cells.InsertNextCell(p)

polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetLines(cells)
polydata.Modified()

if vtk.VTK_MAJOR_VERSION <= 5:
   polydata.Update()

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(input_csv+".vtp");

if vtk.VTK_MAJOR_VERSION <= 5:
   writer.SetInput(polydata)
else:
   writer.SetInputData(polydata)
   writer.Write()
