import cGliderData as cGD

# load data
gdat = cGD.GliderData('LakeSuperior_sci.csv')
gdat.set_timezone('America/Matamoros')
gdat.set_date_range(min_t="2019-06-09 18:00", max_t="2019-06-12 18:00")

# make plots
# gdat.plot('sci_water_pressure')
gdat.plot_xy('sci_water_temp', s=12, c_range=(2.9, 3.3))
gdat.plot_xy('sci_water_temp', contour=True, c_range=(2.9, 3.3))
