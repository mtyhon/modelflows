from modal import Image, Stub, wsgi_app, asgi_app, enter, method
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware


image = Image.from_registry("mtyhon0/modelflows:release1")
stub = Stub("modelflows")

@stub.cls(image=image, gpu="T4") #  checkpointing_enabled=True 
class Initializer():
    
    @enter()    
    def initialize(self):
        import numpy as np
        import torch, zuko
        import plotly.graph_objects as go
        from torch.optim.lr_scheduler import ReduceLROnPlateau

             
        def load_data():
            import os, torch
            import pandas as pd
            from joblib import load

            os.chdir("/app")

            loadpath = 'flow.checkpoint'
            checkpoint = torch.load(loadpath)

            seis_model_df = pd.read_csv('seis_models.csv')
            grid_df = pd.read_csv('MIST_models.csv')

            teff_scaler = load('teff.scaler')
            d01_scaler = load('d01.scaler')
            mms_shell_scaler = load('mms_shell.scaler')
            mms_core_scaler = load('mms_core.scaler')

            return grid_df, seis_model_df, teff_scaler, d01_scaler, mms_shell_scaler, mms_core_scaler, checkpoint

        def infer_samples(flow, cvar, num_samples, device):
            with torch.no_grad():
                ss2, ss_logprobs = flow(torch.Tensor(cvar).to(device)).rsample_and_log_prob((num_samples,))
                ss_logprobs = ss_logprobs.data.cpu().numpy().squeeze()
                ss2 = (ss2.data.cpu().numpy().squeeze())[ss_logprobs > np.percentile(ss_logprobs, 5)]

            return ss2


        grid_df, seis_model_df, teff_scaler, d01_scaler, mms_shell_scaler, mms_core_scaler, checkpoint = load_data()
        

        downsample = 10
        num_marginals = 10000
        num_samples = 1
        fehvar = -1.9
        massvar = 1.
        heliumvar = 0.277
        alfvar = 1.9
        ovshellvar = 0.5
        ovcorevar = 0.5

        main_color = '#A9C0D3'
        seismic_highlight_color = '#E66100'
        hr_highlight_color = '#5D3A9B'
        textbox_color = '#e0e0e1'

        marker_properties = dict(
            size=5,
            color=main_color, line=dict(width=0.5)
        )    

        cv = np.ones((num_marginals, 6)) # mass ,log10(Z), Y, alpha
        cv[:,0] = massvar
        cv[:,1] = fehvar
        cv[:,2] = heliumvar
        cv[:,3] = alfvar
        cv[:,4] = ovshellvar
        cv[:,5] = ovcorevar

        device = torch.device('cuda')
        flow = zuko.flows.NSF(features=9,  context=6, transforms=10, hidden_features=[256] * 10).to(device)
        optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr = 1E-12)

        scheduler.load_state_dict(checkpoint['scheduler'])
        flow.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint (epoch {} - loss {})" .format(checkpoint['epoch'], checkpoint['loss']))
        
        cvar = torch.Tensor(cv).to(device)
        ss2 = infer_samples(flow, cvar, num_samples, device)

        seis_scatter_plot_flow = go.Scattergl(
            x=10**ss2[:,1],
            y=10**ss2[:,3],
            mode='markers',
            marker=marker_properties,
            name='Flow',  
            selected=dict(marker=dict(color=seismic_highlight_color)),  
            customdata=np.vstack([10**ss2[:,1], 10**ss2[:,3], 
                                  10**teff_scaler.inverse_transform(ss2[:,0].reshape(-1,1)).squeeze(),
                                  10**ss2[:,-2], ss2[:,5], 10**ss2[:,-1]]).T
        )

        # dnu, d02, teff, radius, eps, age as customdata to allow same highlighting when switch plots

        seis_cd_plot_model = go.Scattergl(
            x=seis_model_df.dnu,
            y=seis_model_df.d02,
            mode='lines', line=dict(color=f'rgba(169,169,169,0.2)')
        )

        seis_dnueps_plot_model = go.Scattergl(
            x=seis_model_df.eps,
            y=seis_model_df.dnu,
            mode='lines', line=dict(color=f'rgba(169,169,169,0.2)')
        )


        hr_scatter_plot_model = go.Scattergl(
            x=grid_df.teff,
            y=grid_df.rad,
            mode='lines', line=dict(color=f'rgba(169,169,169,0.2)')
        )


        hr_scatter_plot_flow = go.Scattergl(
            x=10**teff_scaler.inverse_transform(ss2[:,0].reshape(-1,1)).squeeze(),
            y=10**ss2[:,-2],
            mode='markers',
            marker=marker_properties,
            name='Flow',
            selected=dict(marker=dict(color=hr_highlight_color)),  # Color for selected 'Flow' points

            customdata=np.vstack([10**ss2[:,1], 10**ss2[:,3], 
                                  10**teff_scaler.inverse_transform(ss2[:,0].reshape(-1,1)).squeeze(),
                                  10**ss2[:,-2], ss2[:,5], 10**ss2[:,-1]]).T
        )

        self.grid_df = grid_df
        self.seis_model_df = seis_model_df
        self.teff_scaler = teff_scaler
        self.d01_scaler = d01_scaler
        self.mms_shell_scaler = mms_shell_scaler
        self.mms_core_scaler = mms_core_scaler

        self.seis_scatter_plot_flow = seis_scatter_plot_flow
        self.hr_scatter_plot_flow = hr_scatter_plot_flow
        self.seis_cd_plot_model = seis_cd_plot_model
        self.seis_dnueps_plot_model = seis_dnueps_plot_model
        self.hr_scatter_plot_model = hr_scatter_plot_model
        self.flow = flow
        
    @method()
    def run(self):
        return self.grid_df, self.seis_model_df,self.teff_scaler,self.d01_scaler,self.mms_shell_scaler,self.mms_core_scaler,self.seis_scatter_plot_flow,self.hr_scatter_plot_flow,self.seis_cd_plot_model,self.seis_dnueps_plot_model,self.hr_scatter_plot_model, self.flow
        

@stub.function(image=image, gpu="T4") # , container_idle_timeout=300, concurrency_limit=1
@asgi_app()
def inyou():

    import numpy as np
    import torch, zuko
    import time as timer
    import plotly.graph_objects as go
    import dash_bootstrap_components as dbc
    import dash
    
    from dash import dcc, html, Input, Output, State

    
    device = torch.device('cuda')
    
    def infer_samples(flow, cvar, num_samples, device):
        with torch.no_grad():
            ss2, ss_logprobs = flow(torch.Tensor(cvar).to(device)).rsample_and_log_prob((num_samples,))
            ss_logprobs = ss_logprobs.data.cpu().numpy().squeeze()
            ss2 = (ss2.data.cpu().numpy().squeeze())[ss_logprobs > np.percentile(ss_logprobs, 5)]

        return ss2
   
 
    
    ########################################## INITIALIZATION ##########################################

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
    app.title = 'Model Flows In You'
    
    init_data = Initializer().run.remote()
    grid_df = init_data[0]
    seis_model_df = init_data[1]
    teff_scaler = init_data[2]
    d01_scaler = init_data[3]
    mms_shell_scaler = init_data[4]
    mms_core_scaler = init_data[5]
    seis_scatter_plot_flow = init_data[6]
    hr_scatter_plot_flow = init_data[7]
    seis_cd_plot_model = init_data[8]
    seis_dnueps_plot_model = init_data[9]
    hr_scatter_plot_model = init_data[10]
    flow = init_data[11]
   

    ########################################## SCATTER LAYOUT ##########################################


    main_color = '#A9C0D3'
    seismic_highlight_color = '#E66100'
    hr_highlight_color = '#5D3A9B'
    textbox_color = '#e0e0e1'
    min_mass, max_mass = 0.70027, 2.49995
    min_z, max_z = -4.934047, -1.29146
    min_y, max_y = 0.23001282, 0.36998718
    min_alpha, max_alpha = 1.000259, 2.699948

    marker_properties = dict(
        size=5,
        color=main_color, line=dict(width=0.5)
    )


    num_marginals = 10000
    num_samples = 1
    fehvar = -1.9
    massvar = 1.
    heliumvar = 0.277
    alfvar = 1.9
    ovshellvar = 0.5
    ovcorevar = 0.5
    
    
    # Define ranges
    cd_xrange = [7.22, 243.2]
    cd_yrange = [-0.72, 21.9]

    dnueps_xrange = [0.32, 2.51]
    dnueps_yrange = cd_xrange

    hr_xrange = [15000, 3000]
    hr_yrange = [0.5, 5.6]


    cd_layout = go.Layout(yaxis=dict(title='δ02 (μHz)',
                                       range=cd_yrange),
                            xaxis=dict(title='Δν (μHz)' 
                                       ,range=cd_xrange, automargin=True),
                            transition_duration=100, height=600,  # Set the height of the figure
                                                                              margin=dict(l=40, r=40, t=10, b=20),
                          dragmode='select',
    #                        uirevision=True,
                              showlegend=False,  # This line hides the legend
                         )

    dnueps_layout = go.Layout(yaxis=dict(title='Δν (μHz)',
                                       range=dnueps_yrange),
                            xaxis=dict(title='ε' 
                                       ,range=dnueps_xrange, automargin=True),
                            transition_duration=100, height=600,  
                            margin=dict(l=40, r=40, t=10, b=20),
                          dragmode='select',
    #                        uirevision=True,
                                  showlegend=False,  
                             )


    hr_layout = go.Layout(yaxis=dict(title='Radius (R⊙)',
                                       range=np.log10(hr_yrange),  type='log'),
                            xaxis=dict(title='Effective Temperature (K)' ,range=hr_xrange, automargin=True),
                          transition_duration=100,height=600,  
                         margin=dict(l=40, r=40, t=10, b=20),
                            dragmode='select',
    #                        uirevision=True,
                              showlegend=False,  
                         )





    ########################################## DIV LAYOUT ##########################################


    slider_width = '42.8%'
    slider_tooltip_always_visible = False
    scatterplot_width = '46.5%'

    # Seismic plot + HR plot with Dropdown
    app.layout = html.Div([
        html.Div([  # Create a div for sliders in two columns
            html.Div([
                html.Label('Mass (M⊙)'),
                dcc.RangeSlider(
                    id='slider_mass',
                    min=min_mass,
                    max=max_mass,
                    step=0.01,
                    value=[massvar, massvar], 
                    allowCross=False,
                    marks ={i: str(i) for i in np.round(np.linspace(min_mass, max_mass, 7 ), 2)},
                    tooltip={"placement": "bottom", "always_visible": slider_tooltip_always_visible}
                )
            ], style={'display': 'inline-block', 'width': slider_width, 
                      'margin-left': '1%', 'margin-right': '5%'}),  # Adjust the width for first column
            html.Div([
                html.Label('Initial Metal Fraction (log10 Z)'),
                dcc.RangeSlider(
                    id='slider_feh',
                    min=min_z,
                    max=max_z,
                    step=0.025,
                    value=[fehvar, fehvar],
                    allowCross=False,
                    marks ={i: str(i) for i in np.round(np.linspace(min_z, max_z, 7 ), 2)},
                    tooltip={"placement": "bottom", "always_visible": slider_tooltip_always_visible}
                )
            ], style={'display': 'inline-block', 'width': slider_width,
                      'margin-left': '4%', 'margin-right': '2%' }),  # Adjust the width for second column

        ], style={'textAlign': 'center', 'margin-bottom': '3px'}),  # Center-align the sliders

        html.Div([  # Create a div for the second row of sliders
            html.Div([
                html.Label('Initial Helium Fraction (Y)'),
                dcc.RangeSlider(
                    id='slider_y',
                    min=min_y,
                    max=max_y,
                    step=0.0025,
                    value=[heliumvar, heliumvar],
                    allowCross=False,
                    marks ={i: str(i) for i in np.round(np.linspace(min_y, max_y, 7 ), 2)},
                    tooltip={"placement": "bottom", "always_visible": slider_tooltip_always_visible}
                )
            ], style={'display': 'inline-block', 'width': slider_width,
                      'margin-left': '1%', 'margin-right': '5%'}),  # Adjust the width for third column
            html.Div([
                html.Label('Mixing Length Parameter (α)'),
                dcc.RangeSlider(
                    id='slider_alpha',
                    min=min_alpha,
                    max=max_alpha,
                    step=0.01,
                    value= [alfvar,alfvar],
                    allowCross=False,
                    marks ={i: str(i) for i in np.round(np.linspace(min_alpha, max_alpha, 9 ), 2)},
                    tooltip={"placement": "bottom", "always_visible": slider_tooltip_always_visible}
                )
            ], style={'display': 'inline-block', 'width': slider_width,
                     'margin-left': '4%', 'margin-right': '2%' }),  # Adjust the width for fourth column
        ], style={'textAlign': 'center', 'margin-bottom': '3px'}),  # Center-align the sliders

        html.Div( [  # Create a div for the first scatter plot with Dropdown
            dcc.Dropdown(
                id='seismic-scatter-dropdown',
                options=[
                    {'label': 'C-D diagram', 'value': 'cd'},
                    {'label': 'Δν-ε diagram', 'value': 'dnueps'},
                    {'label': 'H-R diagram', 'value': 'hr'},
                ],
                value='cd',  # Default option,
                style={'width': '98.5%', 'margin-left': '0.5%', 'margin-right': '1%'}
            ),
            dcc.Store(id='seismic-store', data={'variable': 'cd'}),
            dcc.Store(id='seismic-highlight', data=dict(index=[])),
            dcc.Store(id='last-update-store', data={'variable': None}),
            dcc.Graph(
                id='seismicdiagram',
                figure={
                    'data': [ seis_scatter_plot_flow, seis_cd_plot_model],
                    'layout': cd_layout
                }
            ),
        html.Div(id='seismic-range-textbox', style={'margin-top': '30px', 'margin-left': '10px', 'color': seismic_highlight_color, 
                               'background-color': textbox_color, 'padding': '10px',
                                                   'fontWeight': 'bold', 'border-radius': '10px', 'margin-right': '2%'}), # seismic text box
        ], style={'display': 'inline-block', 'width': scatterplot_width, 'margin-right': '5%', 'margin-left': '1%'} ),


        html.Div([  # Create a div for the second scatter plot
            dcc.Dropdown(
                id='hr-scatter-dropdown',
                options=[
                    {'label': 'H-R diagram', 'value': 'hr'},
                    {'label': 'C-D diagram', 'value': 'cd'},
                    {'label': 'Δν-ε diagram', 'value': 'dnueps'},
                ],
                value='hr',  # Default option
                style={'width': '98.5%', 'margin-left': '0.5%', 'margin-right': '1%'}
            ),
            dcc.Store(id='hr-store', data={'variable': 'hr'}),
            dcc.Store(id='hr-highlight', data=dict(index=[])),
            dcc.Graph(
                id='hrdiagram',
                figure={
                    'data': [hr_scatter_plot_flow, hr_scatter_plot_model],
                    'layout': hr_layout
                }
            ),
        html.Div(id='hr-range-textbox',
                        style={'margin-top': '30px', 'margin-left': '10px', 'color': hr_highlight_color, 
                               'background-color': textbox_color, 'padding': '10px',
                              'fontWeight': 'bold', 'border-radius': '10px', 'margin-right': '2%'}), # hr text box
        ], style={'display': 'inline-block', 'width': scatterplot_width, 'margin-right': '1%'}),
    ])

    ########################################## CALLBACKS ##########################################

    # Callback to update seismic plot variables
    @app.callback(
        Output('seismic-store', 'data'),
        Input('seismic-scatter-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_seismic_variable(code):
        # Code is persistent, defines which variables to plot
        return {'variable': code}



    @app.callback(
        Output('hr-store', 'data'),
        Input('hr-scatter-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_hr_variable(code):
        # Code is persistent, defines which variables to plot
        return {'variable': code}



    @app.callback(
        [Output('seismicdiagram', 'figure'),
         Output('hrdiagram', 'figure'), 
         Output('last-update-store', 'data')],
        [State('seismicdiagram', 'figure'),
         State('hrdiagram', 'figure'), 
         Input('seismic-store', 'data'),
         Input('hr-store', 'data'),
         Input('slider_mass', 'value'),
         Input('slider_feh', 'value'),
         Input('slider_y', 'value'),
         Input('slider_alpha', 'value'),
        Input('seismicdiagram', 'selectedData'),
        Input('hrdiagram', 'selectedData'), 
        State('last-update-store', 'data'),
        State('seismic-highlight', 'data'),
        State('hr-highlight', 'data')],
        prevent_initial_call=True
    )
    def update_all_scatter(seisdiag, hrdiag, codeseis, codehr, newmass, newfeh, newy, newalpha, 
                           seisselect, hrselect, lastupdate, seishighlight, hrhighlight):
        ctx = dash.callback_context
        print('TRIGGERED: ', ctx.triggered_id)

        codehr = codehr['variable']
        codeseis = codeseis['variable']


        layout_dict = {'cd': cd_layout,
                      'dnueps': dnueps_layout,
                      'hr': hr_layout}


        if ctx.triggered_id == None: # Important to keep, otherwise trivial triggers will screw up plot ordering
            return dash.no_update

        elif 'slider' in ctx.triggered_id: # Regenerate predictions across new range and reset selections
             
            cv = np.ones((num_marginals, 6)) # mass ,log10(Z), Y, alpha
            cv[:,0] = np.random.uniform(low=newmass[0], high=newmass[1], size=num_marginals)
            cv[:,1] = np.random.uniform(low=newfeh[0], high=newfeh[1], size=num_marginals)
            cv[:,2] = np.random.uniform(low=newy[0], high=newy[1], size=num_marginals)
            cv[:,3] = np.random.uniform(low=newalpha[0], high=newalpha[1], size=num_marginals)
            cv[:,4] = ovshellvar
            cv[:,5] = ovcorevar

            init_time = timer.time()
            cvar = torch.Tensor(cv).to(device)
            ss2 = infer_samples(flow, cvar, num_samples, device)

            print('Infer Time: ', timer.time() - init_time)
            
            flow_update_dict = {'cd': (10**ss2[:,1], 10**ss2[:,3], cd_layout),
                                'dnueps': (ss2[:,5], 10**ss2[:,1], dnueps_layout),
                                'hr': (10**teff_scaler.inverse_transform(ss2[:,0].reshape(-1,1)).squeeze(),
                                       10**ss2[:,-2], hr_layout )}
            
            newseisdiag = dict(data = [go.Scattergl(
                x=flow_update_dict[codeseis][0],
                y=flow_update_dict[codeseis][1],
                mode='markers',
                marker=marker_properties,
                name='Flow',  
                selected=dict(marker=dict(color=seismic_highlight_color)),  
                unselected=dict(marker=dict(opacity=1)),
                customdata=np.vstack([10**ss2[:,1], 10**ss2[:,3], 
                                      10**teff_scaler.inverse_transform(ss2[:,0].reshape(-1,1)).squeeze(),
                                      10**ss2[:,-2], ss2[:,5], 10**ss2[:,-1]]).T), seisdiag['data'][1]],
                           layout = flow_update_dict[codeseis][2])
            
            newhrdiag = dict(data = [go.Scattergl(
                x=flow_update_dict[codehr][0],
                y=flow_update_dict[codehr][1],
                mode='markers',
                marker=marker_properties,
                name='Flow',  
                selected=dict(marker=dict(color=hr_highlight_color)),  
                unselected=dict(marker=dict(opacity=1)),
                customdata=np.vstack([10**ss2[:,1], 10**ss2[:,3], 
                                      10**teff_scaler.inverse_transform(ss2[:,0].reshape(-1,1)).squeeze(),
                                      10**ss2[:,-2], ss2[:,5], 10**ss2[:,-1]]).T), hrdiag['data'][1]],
                           layout = flow_update_dict[codehr][2])
 

            return newseisdiag, newhrdiag, dict({'variable': ctx.triggered_id})
        

        elif ctx.triggered_id == 'seismic-store':  # Change Plot for Column 1

            flow_update_dict = {'cd': (np.array(seisdiag['data'][0]['customdata'])[:,0],
                                       np.array(seisdiag['data'][0]['customdata'])[:,1], cd_layout,
                                      seis_cd_plot_model),
                    'dnueps': (np.array(seisdiag['data'][0]['customdata'])[:,4], 
                               np.array(seisdiag['data'][0]['customdata'])[:,0], dnueps_layout,
                              seis_dnueps_plot_model),
                    'hr': (np.array(seisdiag['data'][0]['customdata'])[:,2],np.array(seisdiag['data'][0]['customdata'])[:,3],
                           hr_layout, hr_scatter_plot_model)}
            
            seisdiag['data'][0]['x'] = flow_update_dict[codeseis][0]
            seisdiag['data'][0]['y'] = flow_update_dict[codeseis][1]
            seisdiag['data'][1]['x'] = flow_update_dict[codeseis][3].x
            seisdiag['data'][1]['y'] = flow_update_dict[codeseis][3].y
            seisdiag['layout'] = flow_update_dict[codeseis][2]
            
            
            return seisdiag, hrdiag, dict({'variable': ctx.triggered_id})

         
         
        elif ctx.triggered_id == 'hr-store':  # Change Plot for Column 2
         
            flow_update_dict = {'cd': (np.array(hrdiag['data'][0]['customdata'])[:,0], 
                                       np.array(hrdiag['data'][0]['customdata'])[:,1], cd_layout,
                                      seis_cd_plot_model),
                    'dnueps': (np.array(hrdiag['data'][0]['customdata'])[:,4],
                               np.array(hrdiag['data'][0]['customdata'])[:,0], dnueps_layout,
                              seis_dnueps_plot_model),
                    'hr': (np.array(hrdiag['data'][0]['customdata'])[:,2], 
                           np.array(hrdiag['data'][0]['customdata'])[:,3], hr_layout, hr_scatter_plot_model )}

            hrdiag['data'][0]['x'] = flow_update_dict[codehr][0]
            hrdiag['data'][0]['y']=  flow_update_dict[codehr][1]
            hrdiag['data'][1]['x'] = flow_update_dict[codehr][3].x
            hrdiag['data'][1]['y'] = flow_update_dict[codehr][3].y
            hrdiag['layout'] = flow_update_dict[codehr][2]

            return seisdiag, hrdiag, dict({'variable': ctx.triggered_id})


        elif ctx.triggered_id == 'seismicdiagram': 

            if (seisselect == None) :# 
                seis_selected_indices = []
            elif (seisselect['points'] == []) & (lastupdate['variable'] == 'seismicdiagram'):
                return dash.no_update
            else:
                seis_selected_indices = [point['pointIndex'] for point in seisselect['points'] if point['curveNumber'] == 0] 

            if len(np.unique(seisdiag['data'][0]['marker']['color'])) == 1:
                colorz = [seismic_highlight_color if i in seis_selected_indices else main_color for i in range(len(seisdiag['data'][0]['x']))]                
            elif seisselect != None:
                colorz = [seismic_highlight_color if i in seis_selected_indices else hr_highlight_color if hrdiag['data'][0]['marker']['color'][i] == hr_highlight_color else main_color for i in range(len(hrdiag['data'][0]['x']))]
            elif hrselect != None: #hr is monochromatic
                colorz = [seismic_highlight_color if i in seis_selected_indices else hr_highlight_color if i in hrhighlight['index'] else main_color for i in range(len(hrdiag['data'][0]['x']))]
            else:
                colorz = main_color
                
            flow_update_dict = {'cd': cd_layout,
                    'dnueps':  dnueps_layout,
                    'hr': hr_layout }
            
            newseisdiag = dict(data = [go.Scattergl(
                x=seisdiag['data'][0]['x'],
                y=seisdiag['data'][0]['y'],
                mode='markers',
                marker=marker_properties,
                name='Flow',  
                selected=dict(marker=dict(color=seismic_highlight_color)),  
                unselected=dict(marker=dict(opacity=1)),
                customdata=seisdiag['data'][0]['customdata']), seisdiag['data'][1]],
                           layout = flow_update_dict[codeseis])
            
            newhrdiag = dict(data = [go.Scattergl(
                x=hrdiag['data'][0]['x'],
                y=hrdiag['data'][0]['y'],
                mode='markers',
                marker=marker_properties,
                name='Flow',  
                selected=dict(marker=dict(color=hr_highlight_color)),  
                unselected=dict(marker=dict(opacity=1)),
                customdata=hrdiag['data'][0]['customdata']), hrdiag['data'][1]],
                           layout = flow_update_dict[codehr])
                   
            
            newseisdiag['data'][0]['marker']['color'] = colorz
            newhrdiag['data'][0]['marker']['color'] = colorz

            
            return newseisdiag, newhrdiag, dict({'variable': ctx.triggered_id})



        elif ctx.triggered_id == 'hrdiagram':  

            if (hrselect == None):
                hr_selected_indices = []
            elif (hrselect['points'] == []) & (lastupdate['variable'] == 'hrdiagram'):
                return dash.no_update
            else:
                hr_selected_indices = [point['pointIndex'] for point in hrselect['points'] if point['curveNumber'] == 0] 
            
            if len(np.unique(hrdiag['data'][0]['marker']['color'])) == 1:
                colorz = [hr_highlight_color if i in hr_selected_indices else main_color for i in range(len(hrdiag['data'][0]['x']))]   
            elif hrselect != None:
                colorz = [hr_highlight_color if i in hr_selected_indices else seismic_highlight_color if seisdiag['data'][0]['marker']['color'][i] == seismic_highlight_color else main_color for i in range(len(seisdiag['data'][0]['x']))]
            elif seisselect != None:
                colorz = [hr_highlight_color if i in hr_selected_indices else seismic_highlight_color if i in seishighlight['index'] else main_color for i in range(len(seisdiag['data'][0]['x']))]
            else:
                colorz = main_color
            
            flow_update_dict = {'cd': cd_layout,
                    'dnueps':  dnueps_layout,
                    'hr': hr_layout }
            
            newseisdiag = dict(data = [go.Scattergl(
                x=seisdiag['data'][0]['x'],
                y=seisdiag['data'][0]['y'],
                mode='markers',
                marker=marker_properties,
                name='Flow',  
                selected=dict(marker=dict(color=seismic_highlight_color)),  
                unselected=dict(marker=dict(opacity=1)),
                customdata=seisdiag['data'][0]['customdata']), seisdiag['data'][1]],
                           layout = flow_update_dict[codeseis])
            
            newhrdiag = dict(data = [go.Scattergl(
                x=hrdiag['data'][0]['x'],
                y=hrdiag['data'][0]['y'],
                mode='markers',
                marker=marker_properties,
                name='Flow',  
                selected=dict(marker=dict(color=hr_highlight_color)),  
                unselected=dict(marker=dict(opacity=1)),
                customdata=hrdiag['data'][0]['customdata']), hrdiag['data'][1]],
                           layout = flow_update_dict[codehr])
                   
            
            newseisdiag['data'][0]['marker']['color'] = colorz
            newhrdiag['data'][0]['marker']['color'] = colorz

            return newseisdiag, newhrdiag, dict({'variable': ctx.triggered_id})

        else:
            return dash.no_update

    @app.callback(
        Output('seismic-range-textbox', 'children'),
        [State('seismicdiagram', 'figure'),
         Input('seismicdiagram', 'selectedData'),     
         Input('slider_mass', 'value'),
         Input('slider_feh', 'value'),
         Input('slider_y', 'value'),
         Input('slider_alpha', 'value')],
        prevent_initial_call=False
    )
    def update_seismic_range_textbox(seisdiag, selected_data, newmass, newfeh, newy, newalpha):
        ctx = dash.callback_context
        if ctx.triggered_id == None: # Important to keep, otherwise trivial triggers will screw up plot ordering
            return 'Selected Age Range: None'
        elif 'slider' in ctx.triggered_id:
            return 'Selected Age Range: None'
        elif selected_data is None or 'points' not in selected_data:
            return 'Selected Age Range: None'

        seis_selected_indices = [point['pointIndex'] for point in selected_data['points'] if point['curveNumber'] == 0]
        if len(seis_selected_indices) == 0:
            return dash.no_update
        ageval = np.array(seisdiag['data'][0]['customdata'])[:,-1][np.array(seis_selected_indices)]

        return 'Selected Range: %.3f - %.3f Gyr' %(min(ageval), max(ageval))


    @app.callback(
        Output('hr-range-textbox', 'children'),
        [State('hrdiagram', 'figure'),
         Input('hrdiagram', 'selectedData'),
        Input('slider_mass', 'value'),
         Input('slider_feh', 'value'),
         Input('slider_y', 'value'),
         Input('slider_alpha', 'value')],
        prevent_initial_call=False
    )
    def update_hr_range_textbox(hrdiag, selected_data, newmass, newfeh, newy, newalpha):
        ctx = dash.callback_context
        if ctx.triggered_id == None: # Important to keep, otherwise trivial triggers will screw up plot ordering
            return 'Selected Age Range: None'
        elif 'slider' in ctx.triggered_id:
            return 'Selected Age Range: None'
        elif selected_data is None or 'points' not in selected_data:
            return 'Selected Age Range: None'

        hr_selected_indices = [point['pointIndex'] for point in selected_data['points'] if point['curveNumber'] == 0]
        if len(hr_selected_indices) == 0:
            return dash.no_update
        ageval = np.array(hrdiag['data'][0]['customdata'])[:,-1][np.array(hr_selected_indices)]

        return 'Selected Age Range: %.3f - %.3f Gyr' %(min(ageval), max(ageval))
    
    @app.callback(
        Output('seismic-highlight', 'data'),
        [Input('seismicdiagram', 'selectedData'),
        State('seismic-highlight', 'data'),
        State('last-update-store', 'data')],
        prevent_initial_call=False
    )
    def update_seis_highlight(seisselect, seishighlight, lastupdate):
        if (seisselect == None):
            seis_selected_indices = seishighlight['index']
        elif (seisselect['points'] == []) & (lastupdate['variable'] == 'seismicdiagram'):
            return dash.no_update
        else:
            seis_selected_indices = [point['pointIndex'] for point in seisselect['points'] if point['curveNumber'] == 0] 
        
        return {'index': seis_selected_indices}

    @app.callback(
        Output('hr-highlight', 'data'),
        [Input('hrdiagram', 'selectedData'),
        State('hr-highlight', 'data'),
        State('last-update-store', 'data')],
        prevent_initial_call=False
    )
    def update_hr_highlight(hrselect, hrhighlight, lastupdate):
        if (hrselect == None):
            hr_selected_indices = hrhighlight['index']
        elif (hrselect['points'] == []) & (lastupdate['variable'] == 'hrdiagram'):
            return dash.no_update
        else:
            hr_selected_indices = [point['pointIndex'] for point in hrselect['points'] if point['curveNumber'] == 0] 
        
        return {'index': hr_selected_indices}
    
    server = FastAPI()
    server.mount("/", WSGIMiddleware(app.server))
    
    return server
    

