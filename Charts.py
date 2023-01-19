# Colors
# https://plotly.com/python-api-reference/generated/plotly.express.colors.html

# color_scale = px.colors.sequential.RdBu # https://plotly.com/python/builtin-colorscales/
# color_scale = px.colors.qualitative.Light24 # https://plotly.com/python/discrete-color/

from datetime import datetime as dt
import inspect
import numpy as np
import pandas as pd
import statsmodels.api as sm

import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode, JsCode


def chart_heat_map(heat_map_df, x_col,y_col,z_col,range_color=None, add_mean=False, sort_by=None, abs=False, subtract=None, simmetric_sort=False, transpose=False, drop_cols=[], color_continuous_scale='RdBu', format_labels=None, title=None,tickangle=None, sorted_cols=[]):
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
        if (('_abs' in sort_by) & (sort_by not in heat_map.columns)):
            sort_var=sort_by.split('_')[0]
            heat_map[sort_by]=heat_map[sort_var].abs()
            heat_map=heat_map.sort_values(by=sort_by, ascending=False)
            heat_map=heat_map.drop(columns=[sort_by])
        else:
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

    if len(sorted_cols)>0:
        heat_map=heat_map[sorted_cols]
    fig = px.imshow(heat_map, color_continuous_scale=color_continuous_scale, range_color=range_color,title=title, aspect='auto')

    if format_labels is not None:
        fig.update_traces(texttemplate=format_labels)

    fig.update_yaxes(dtick=1,tickangle=tickangle,automargin=True,title=None)
    fig.update_xaxes(dtick=1,tickangle=tickangle,automargin=True,title=None)

    return fig

def scatter_matrix_chart(df, marker_color='blue', add_trendline=True, add_title=True, vertical_spacing=0.03, horizontal_spacing=0.01, marker_size=2, today_index=None, today_size=5, prediction_index=None, prediction_size=5, x_tickangle=90, y_tickangle=0):
    cols=list(df.columns)

    if add_title:
        titles=['title ' + str(i) for i in range(len(cols)*len(cols))]
    else:
        titles=[]

    fig = make_subplots(rows=len(cols), cols=len(cols), shared_xaxes=True, shared_yaxes=True, subplot_titles=titles, vertical_spacing=vertical_spacing, horizontal_spacing=horizontal_spacing)
    mode='markers'
    
    anno_count=0
    for ri, yc in enumerate(cols):
        for ci, xc in enumerate(cols):
            rr=ri+1
            cc=ci+1

            x=df[xc]
            y=df[yc]

            date_format = "%d %B %Y"
            y_str = 'Y: '+ yc +' %{y:.2f}'
            x_str = 'X: '+ xc +' %{x:.2f}'
            text=[]
            if xc=='date':
                text = [d.strftime(date_format) for d in [dt.fromordinal(i) for i in x]]
                x_str='X: %{text}'

            if yc=='date':
                text = [d.strftime(date_format) for d in [dt.fromordinal(i) for i in y]]
                y_str='Y: %{text}'

            hovertemplate="<br>".join([y_str, x_str, "<extra></extra>"])

            fig.add_trace(go.Scatter(x=x, y=y, mode=mode,marker=dict(size=marker_size,color=marker_color),hovertemplate=hovertemplate,text=text), row=rr, col=cc)
            if today_index is not None:
                add_today(fig,df,xc,yc,today_index,today_size, row=rr, col=cc)
                        
            fig.update_xaxes(row=rr, col=cc, showgrid=False,zeroline=False)
            if rr==len(cols):
                tick_pos=(x.max()+x.min())/2.0
                fig.update_xaxes(row=rr, col=cc, tickangle=x_tickangle,automargin=True,tickvals=[tick_pos],ticktext=[xc], showgrid=False,zeroline=False)

            fig.update_yaxes(row=rr, col=cc, showgrid=False,zeroline=False)
            if cc==1:
                tick_pos=(y.max()+y.min())/2.0
                fig.update_yaxes(row=rr, col=cc, tickangle=y_tickangle,automargin=True,tickvals=[tick_pos],ticktext=[yc],showgrid=False,zeroline=False)

            if ((add_trendline) | (add_title)):
                model = sm.OLS(y.values, sm.add_constant(x.values), missing="drop").fit()
                r_sq_str="Rsq "+str(round(100*model.rsquared,1))
                hovertemplate="<br>".join([r_sq_str, "<extra></extra>"])

                if add_trendline:
                    fig.add_trace(go.Scatter(x=x, y=model.predict(), mode='lines',hovertemplate=hovertemplate, line=dict(color='black', width=0.5)), row=rr, col=cc)
                    pred_str=''
                    print('prediction_index',prediction_index)
                    
                    if prediction_index is not None:
                        pred_str= 'Pred '+str(round(add_today(fig,df,xc,yc,prediction_index, size=prediction_size, color='black', symbol='x', name='Prediction', row=rr, col=cc,model=model),1))
                    
                if add_title:
                    fig.layout.annotations[anno_count].update(text=r_sq_str+ ' '+pred_str)
                    anno_count+=1
    
    fig.update_layout(showlegend=False)
    return fig

def sorted_scatter_chart(df, y_col, N_col_subplots=5, marker_color='blue', add_trendline=True, add_title=True, vertical_spacing=0.03, horizontal_spacing=0.01, marker_size=2, today_index=None, today_size=5, prediction_index=None, prediction_size=5, x_tickangle=90, y_tickangle=0):
    """
    N_col_subplots = 5
        - it means: 5 chart in each row
    """
    
    cols=list(df.columns)

    if add_title:
        titles=['title ' + str(i) for i in range(len(cols))]
    else:
        titles=[]

    cols=list(df.columns)
    cols_subsets=[]
    for i in range(0, len(cols), N_col_subplots):
        cols_subsets=cols_subsets+[cols[i:i+N_col_subplots]]

    fig = make_subplots(rows=len(cols_subsets), cols=len(cols_subsets[0]), shared_xaxes=False, shared_yaxes=True, subplot_titles=titles, vertical_spacing=vertical_spacing, horizontal_spacing=horizontal_spacing)

    mode='markers'
    
    anno_count=0
    for ri, cols in enumerate(cols_subsets):
        for ci, xc in enumerate(cols):
            rr=ri+1
            cc=ci+1

            x=df[xc]
            y=df[y_col]

            date_format = "%d %B %Y"
            y_str = 'Y: '+ y_col +' %{y:.2f}'
            x_str = 'X: '+ xc +' %{x:.2f}'
            text=[]
            if xc=='date':
                text = [d.strftime(date_format) for d in [dt.fromordinal(i) for i in x]]
                x_str='X: %{text}'

            if y_col=='date':
                text = [d.strftime(date_format) for d in [dt.fromordinal(i) for i in y]]
                y_str='Y: %{text}'

            hovertemplate="<br>".join([y_str, x_str, "<extra></extra>"])

            fig.add_trace(go.Scatter(x=x, y=y, mode=mode,marker=dict(size=marker_size,color=marker_color),hovertemplate=hovertemplate,text=text), row=rr, col=cc)
            if today_index is not None:
                add_today(fig,df,xc,y_col,today_index,today_size, row=rr, col=cc)
            
            # X-axis
            tick_pos=(x.max()+x.min())/2.0
            fig.update_xaxes(row=rr, col=cc, tickangle=x_tickangle,automargin=True,tickvals=[tick_pos],ticktext=[xc], showgrid=False,zeroline=False)

            # Y-axis
            tick_pos=(y.max()+y.min())/2.0
            fig.update_yaxes(row=rr, col=cc, tickangle=y_tickangle,automargin=True,tickvals=[tick_pos],ticktext=[y_col],showgrid=False,zeroline=False)

            if ((add_trendline) | (add_title)):
                model = sm.OLS(y.values, sm.add_constant(x.values), missing="drop").fit()
                r_sq_str="Rsq "+str(round(100*model.rsquared,1))
                hovertemplate="<br>".join([r_sq_str, "<extra></extra>"])

                if add_trendline:
                    fig.add_trace(go.Scatter(x=x, y=model.predict(), mode='lines',hovertemplate=hovertemplate, line=dict(color='black', width=0.5)), row=rr, col=cc)
                    pred_str=''
                    print('prediction_index',prediction_index)
                    
                    if prediction_index is not None:
                        pred_str= 'Pred '+str(round(add_today(fig,df,xc,y_col,prediction_index, size=prediction_size, color='black', symbol='x', name='Prediction', row=rr, col=cc,model=model),1))
                    
                if add_title:
                    fig.layout.annotations[anno_count].update(text=r_sq_str+ ' '+pred_str)
                    anno_count+=1
    
    fig.update_layout(showlegend=False)
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
    colorscale_dict['Red-only']=['red','red']
    colorscale_dict['Blue-only']=['blue','blue']
    colorscale_dict['Green-only']=['green','green']
    colorscale_dict['Black-only']=['black','black']
    return colorscale_dict

def add_today(fig, df, x_col, y_col, today_idx, size=10, color='red', symbol='star', name='Today', model=None, row=1, col=1):
    """
    if 'model' is not None, it will calculate the prediction
    markers:
        https://plotly.com/python/marker-style/
    """

    x = df.loc[today_idx][x_col]

    if model is None:    
        y = df.loc[today_idx][y_col]
    else:
        pred_df=sm.add_constant(df, has_constant='add').loc[today_idx][['const',x_col]]
        y=model.predict(pred_df)[0]
    
    y_str = 'Y: '+ y_col +' %{y:.2f}'
    x_str = 'X: '+ x_col +' %{x:.2f}'
    hovertemplate="<br>".join([name, y_str, x_str, "<extra></extra>"])
    fig.add_trace(go.Scatter(name=name,x=[x], y=[y], mode = 'markers', marker_symbol = symbol,marker_size = size, marker_color=color, hovertemplate=hovertemplate), row=row, col=col)
    return y

def aggrid_var_search(df,rows_per_page=None, pre_selected_rows=[]):
    # Decide which columns to show
    visible_cols=list(df.columns)
    hide_cols=list(set(df.columns)-set(visible_cols))

    # Sort the columns
    sort_cols=visible_cols
    sort_cols.extend(hide_cols)
    df=df[sort_cols]

    if rows_per_page is None:
        rows_per_page=len(df)

    statusPanels = {'statusPanels': [
    { 'statusPanel': 'agFilteredRowCountComponent', 'align': 'left' },
    { 'statusPanel': 'agSelectedRowCountComponent', 'align': 'left' },
    { 'statusPanel': 'agAggregationComponent', 'align': 'left' },
    ]}

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=rows_per_page)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True, rowMultiSelectWithClick=False, floatingFilter=True)
    gb.configure_selection('multiple', use_checkbox=True, pre_selected_rows=pre_selected_rows)
    gb.configure_grid_options(enableRangeSelection=False, statusBar=statusPanels)
    # gb.configure_side_bar(defaultToolPanel='test')    

    # Single columns configuration
    gb.configure_column('Selection', headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly=True, filter= 'agSetColumnFilter', suppressMenu=True, filterParams=dict (excelMode= 'windows'))

    # for h in hide_cols:
    gb.configure_columns(hide_cols, hide = True)

    # good
    gridOptions = gb.build()
    grid_response = AgGrid(df, gridOptions=gridOptions, 
                        data_return_mode=DataReturnMode.FILTERED,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                        reload_data=False,enable_enterprise_modules=True, allow_unsafe_jscode=True)
   
    return grid_response

def aggrid_var_selected(df,rows_per_page=None, pre_selected_rows=[]):
    # Decide which columns to show
    visible_cols=list(df.columns)
    hide_cols=list(set(df.columns)-set(visible_cols))

    # Sort the columns
    sort_cols=visible_cols
    sort_cols.extend(hide_cols)
    df=df[sort_cols]

    if rows_per_page is None:
        rows_per_page=len(df)

    statusPanels = {'statusPanels': [
    { 'statusPanel': 'agFilteredRowCountComponent', 'align': 'left' },
    { 'statusPanel': 'agSelectedRowCountComponent', 'align': 'left' },
    { 'statusPanel': 'agAggregationComponent', 'align': 'left' },
    ]}

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=rows_per_page)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True, rowMultiSelectWithClick=False, floatingFilter=True)
    gb.configure_selection('multiple', use_checkbox=False, pre_selected_rows=pre_selected_rows)
    gb.configure_grid_options(enableRangeSelection=False, statusBar=statusPanels)
    # gb.configure_side_bar(defaultToolPanel='test')    

    # Single columns configuration
    gb.configure_column('Selection', headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly=True, filter= 'agSetColumnFilter', suppressMenu=True, filterParams=dict (excelMode= 'windows'))

    # for h in hide_cols:
    gb.configure_columns(hide_cols, hide = True)

    # good
    gridOptions = gb.build()
    grid_response = AgGrid(df, gridOptions=gridOptions, 
                        data_return_mode=DataReturnMode.FILTERED,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                        reload_data=False,enable_enterprise_modules=True, allow_unsafe_jscode=True)
   
    return grid_response