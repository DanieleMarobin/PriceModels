# Colors
# https://plotly.com/python-api-reference/generated/plotly.express.colors.html

# color_scale = px.colors.sequential.RdBu # https://plotly.com/python/builtin-colorscales/
# color_scale = px.colors.qualitative.Light24 # https://plotly.com/python/discrete-color/

from datetime import datetime as dt
import inspect
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def chart_heat_map(heat_map_df, x_col,y_col,z_col,range_color=None, add_mean=False, sort_by=None, abs=False, subtract=None, simmetric_sort=False, transpose=False, drop_cols=[], color_continuous_scale='RdBu', format_labels=None, title=None,tickangle=None):
    """
        heat_map_df: it must have 3 columns, to be able to have x,y, and values to put into the heat matrix

        'format_labels' example: '%{z:.1f}%'
    """
    # heat_map = heat_map_df.pivot_table(index=[y_col], columns=[x_col], values=[z_col], aggfunc=aggfunc)
    heat_map = heat_map_df.pivot(index=[y_col], columns=[x_col], values=[z_col])    
    heat_map.columns = heat_map.columns.droplevel(level=0)

    if add_mean:
        heat_map['mean']=heat_map.mean(axis=1)

    if sort_by is not None:
        heat_map=heat_map.sort_values(by=sort_by, ascending=False)

    if simmetric_sort:
        sorted_cols = list(heat_map.index)
        
        if add_mean:
            sorted_cols.extend(['mean'])
        
        heat_map=heat_map[sorted_cols]

    if abs:
        heat_map=heat_map.abs()

    if subtract is not None:
        heat_map=heat_map.subtract(heat_map[subtract],axis=0)

    heat_map=heat_map.drop(columns=drop_cols)

    if transpose:
        heat_map=heat_map.T
    # return heat_map

    fig = px.imshow(heat_map, color_continuous_scale=color_continuous_scale, range_color=range_color,title=title, aspect='auto')

    if format_labels is not None:
        fig.update_traces(texttemplate=format_labels)

    fig.update_yaxes(dtick=1,tickangle=tickangle,automargin=True,title=None)
    fig.update_xaxes(dtick=1,tickangle=tickangle,automargin=True,title=None)

    return fig



def get_plotly_colorscales():
    """
    color_scales = uc.get_plotly_colorscales()
    fig=px.scatter(df,x='x',y='y', color_continuous_scale=color_scales[chart_color_key], color_discrete_sequence=color_scales[chart_color_key])
    """    
    colorscale_dict={}
    colors_modules = ['carto', 'cmocean', 'cyclical','diverging', 'plotlyjs', 'qualitative', 'sequential']
    for color_module in colors_modules:
        colorscale_dict.update({name+'-'+color_module:body for name, body in inspect.getmembers(getattr(px.colors, color_module)) if (isinstance(body, list) & ('__all__' not in name))})
        
    return colorscale_dict


def plot_plotly_colorscales(step=0.1, colors_modules = ['carto', 'cmocean', 'cyclical','diverging', 'plotlyjs', 'qualitative', 'sequential']):   
    x=np.arange(1,-1.001,-step)
    y=np.arange(-1,1.001,step)

    matrix=(y.reshape(1, -1) + x.reshape(-1 ,1))

    color_scales=get_plotly_colorscales()
    for k, v in color_scales.items():
        if np.isin( k.split('-')[-1], colors_modules):
            try:
                fig=px.imshow(matrix,title=k,color_continuous_scale=v)
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                # fig.update_coloraxes(showscale=False)
                fig.show('browser')
            except:
                print('Cannot use: '+ k)

    print('Done')