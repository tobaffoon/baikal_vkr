"""
ESMValTool diagnostic for ESA CCI LST data.

The code uses the all time average monthly data.
The ouptput is a timeseries plot of the mean differnce of
CCI LST to model ensemble average, with the ensemble spread
represented by a standard deviation either side of the mean.
"""

import logging

import collections
import iris
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn.linear_model import LinearRegression

from esmvaltool.diag_scripts.shared import (
   ProvenanceLogger,
   get_plot_filename,
   group_metadata,
   run_diagnostic
)

mpl.rcParams['figure.dpi'] = 300

logger = logging.getLogger(__name__)


def _get_input_cubes(metadata):
   """Load the data files into cubes.

   Based on the hydrology diagnostic.

   Inputs:
   metadata = List of dictionaries made from the preprocessor config

   Outputs:
   inputs = Dictionary of cubes
   ancestors = Dictionary of filename information
   """
   inputs = {}
   ancestors = {}
   for attributes in metadata:
      short_name = attributes['short_name']
      filename = attributes['filename']
      logger.info("Loading variable %s", short_name)
      cube = iris.load_cube(filename)
      cube.attributes.clear()
      inputs[short_name] = cube
      ancestors[short_name] = [filename]

   return inputs, ancestors

def _plot_annual_mean_multiple(year_temp_dict_coll, colors, names, plot_save_path, display_coef=False, check_years=True):
   """Create and save the output figure.

   The plot is just mean values of multiple models

   Inputs:
   year_temp_dict_coll - list of lists of years mapped to ts
   colors - list of colors to draw temperature with
   names - names of model

   Outputs:
   Saved figure
   """
   fig, ax = plt.subplots()
   plt.gcf().set_size_inches(10, plt.gcf().get_size_inches()[1])

   # ax.set_title('Baikal')
   # fig.suptitle('ESACCI LST - CMIP6 Historical Ensemble Mean', fontsize=24)
   ax.set_xlabel('Год')
   ax.set_ylabel(r"Температура поверхностного слоя, ${\degree}C$")

   years = list(year_temp_dict_coll[0].keys())
   if check_years:
      for d in year_temp_dict_coll:
         years = np.intersect1d(list(years), list(d.keys()))
   else:
      years = list(set([year for year_temp_dict in year_temp_dict_coll for year in year_temp_dict.keys()]))


   ax.set_xticks(np.array(years))
   ax.tick_params(axis='x', labelrotation=300)

   for i, (yt_dict, name) in enumerate(zip(year_temp_dict_coll, names)):
      tss = [yt_dict[year] for year in years if year in yt_dict]
      ys = [year for year in years if year in yt_dict]
      regression_years = np.array(ys).reshape(-1,1)

      regressor = LinearRegression().fit(regression_years, tss)

      ax.plot(
         ys,
         tss,
         linewidth=1.5,
         color=colors(i),
         # label=f"Данные {name}"
      )

      ax.plot(
         ys,
         regressor.predict(regression_years),
         linewidth=0.5,
         color=colors(i, alpha=0.4),
         # label=f"Линейная регрессия {name}"
         label=f"{name}"
      )

      if display_coef:
         ax.text(
            x=0.815,
            y=0.95 - i*0.1,
            s=f"$\it{{{name}: {regressor.coef_[0]:.2f}}}{{\degree}}C/год$",
            horizontalalignment='center',
            verticalalignment='top',
            transform = ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.2))
      
   ax.legend(loc="upper left", fontsize=6)

   plt.savefig(plot_save_path)

def _plot_annual_mean(yt_dict, name, plot_save_path):
   """Create and save the output figure.

   The plot is just mean values

   Inputs:

   Outputs:
   Saved figure
   """
   fig, ax = plt.subplots()
   plt.gcf().set_size_inches(10, plt.gcf().get_size_inches()[1])

   # ax.set_title('Baikal')
   # fig.suptitle('ESACCI LST - CMIP6 Historical Ensemble Mean', fontsize=24)
   ax.set_xlabel('Год')
   ax.set_ylabel(r"Температура поверхностного слоя, ${\degree}C$")

   years = np.array(list(yt_dict.keys()))
   tss = np.array(list(yt_dict.values()))

   ax.set_xticks(years)
   ax.tick_params(axis='x', labelrotation=300)

   regression_years = np.array(years).reshape(-1,1)
   regressor = LinearRegression().fit(regression_years, tss)

   ax.plot(
      years,
      tss,
      linewidth=1.5,
      color="black",
      label=f"Данные {name}"
   )

   ax.plot(
      years,
      regressor.predict(regression_years),
      linewidth=0.5,
      color="black",
      label=f"Линейная регрессия {name}"
   )

   ax.text(
      x=0.815,
      y=0.95,
      s=f"$\it{{{name}: {regressor.coef_[0]:.2f}}}{{\degree}}C/год$",
      horizontalalignment='center',
      verticalalignment='top',
      transform = ax.transAxes,
      bbox=dict(facecolor='white', alpha=0.2))
   
   ax.legend(loc="upper left")

   plt.savefig(plot_save_path)


def _get_provenance_record(attributes, ancestor_files):
   """Create the provenance record dictionary.

   Inputs:
   attributes = dictionary of ensembles/models used, the region bounds
               and years of data used.
   ancestor_files = list of data files used by the diagnostic.

   Outputs:
   record = dictionary of provenance records.
   """
   caption = "Timeseries"

   record = {
      'caption': caption,
      'statistics': ['mean', 'stddev'],
      'domains': ['reg'],
      'plot_types': ['times'],
      'authors': ['king_robert'],
      # 'references': [],
      'ancestors': ancestor_files
   }

   return record

def _prepare_landsat():
   lst_landsat = { 1996: 7.615046916830711,
 1997: 6.4652446881016505,
 1998: 8.29735257295523,
 1999: 7.466276454270664,
 2000: 8.444648433179971,
 2001: 6.8573912127145835,
 2002: 11.29452174088983,
 2003: 8.133507938438473,
 2004: 7.756460249341295,
 2005: 8.815491983317168,
 2006: 6.796633766422254,
 2007: 9.983745663150753,
 2008: 8.444456981800055,
 2009: 7.979558202903978,
 2010: 7.409646087737466,
 2011: 7.802575171575934,
 2012: 8.998809616805136,
 2013: 9.029895691880107,
 2014: 10.268777774431134,
 2015: 10.722652392625395,
 2016: 10.075703142279956,
 2017: 9.112241223415538,
 2018: 9.1208041147658,
 2019: 9.838905106835822,
 2020: 10.8137280883284,
 2021: 8.059809265577483,
 2022: 8.255228706723798,
 2023: 9.533078486705547,
 2024: 9.525957321098034}
   return collections.OrderedDict(sorted(lst_landsat.items()))

def _diagnostic(config):
   """Perform the control for the ESA CCI LST diagnostic.

   Parameters
   ----------
   config: dict
      the preprocessor nested dictionary holding
      all the needed information.

   Returns
   -------
   figures made by make_plots.
   """
   # this loading function is based on the hydrology diagnostic
   input_metadata = config['input_data'].values()

   yt_dict_landsat = _prepare_landsat()
   two_cm = plt.get_cmap('gist_rainbow', 2)

   ancestor_list = []
   yt_dict_coll = [yt_dict_landsat]
   names = ["landsat"]
   with ProvenanceLogger(config) as provenance_logger:
      for exp, metadata in group_metadata(input_metadata, 'exp').items():
         cubes, ancestors = _get_input_cubes(metadata)
         ts_cube = cubes['ts'] # ts - temperature of surface
         year_ts_dict = _get_average_annual_ts(ts_cube)

         yt_dict_coll.append(year_ts_dict)
         names.append(exp)
         
         # Provenance
         ancestor_list.append(ancestors['ts'][0]) 

         timerange = ts_cube.coord('time').units.num2date(ts_cube.coord('time').points)
         data_attributes = {}
         data_attributes['start_year'] = timerange[0].year
         data_attributes['end_year'] = timerange[-1].year
         data_attributes['ensembles'] = ''

         record = _get_provenance_record(data_attributes, ancestor_list)

         solo_plot_path = get_plot_filename(exp, config)
         landsat_plot_path = get_plot_filename(f"{exp}_Landsat_comparison", config)
         _plot_annual_mean(year_ts_dict, exp, solo_plot_path)
         _plot_annual_mean_multiple([yt_dict_landsat, year_ts_dict], two_cm, ["Landsat", exp], landsat_plot_path)

         provenance_logger.log(solo_plot_path, record)
   
   colors = plt.get_cmap('gist_rainbow', len(yt_dict_coll))
   all_path = get_plot_filename("all_experiments", config)
   all_no_cap_path = get_plot_filename("all_experiments_no_year_limit", config)
   _plot_annual_mean_multiple(yt_dict_coll, colors, names, all_path)
   _plot_annual_mean_multiple(yt_dict_coll, colors, names, all_no_cap_path, check_years=False)

def _get_average_annual_ts(ts_cube: iris.cube):
   years = _get_cube_years(ts_cube)
   ts = {}
   for year in years:
      # calculate mean for each year once
      ts[int(year)] = _get_average_year_ts(ts_cube, year)

   return collections.OrderedDict(sorted(ts.items()))

def _get_cube_years(ts_cube: iris.cube) -> list[int]:
   found_year_coord = len(ts_cube.coords("year"))>0
   if found_year_coord:
      return set(ts_cube.coord("year").points)
   years = set()
   unit = ts_cube.coord("time").units
   for entry in ts_cube.coord("time").points:
      year = unit.num2pydate(entry).year
      years.add(year)
   logger.info(f"YEARS: {years}")
   return list(years)

def _get_average_year_ts(ts_cube: iris.cube, year: int):
   openwater_month_start = 6 # июнь
   openwater_month_end = 11 # ноябрь
   warm_season_constraint = iris.Constraint(time=lambda cell: cell.point.year == year and openwater_month_start <= cell.point.month < openwater_month_end)
   ts_cube = warm_season_constraint.extract(ts_cube)
   return ts_cube.data.mean()

if __name__ == '__main__':
   # always use run_diagnostic() to get the config (the preprocessor
   # nested dictionary holding all the needed information)
   with run_diagnostic() as config:
      _diagnostic(config)