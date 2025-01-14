
from shiny import App, reactive, render, ui

from shinywidgets import render_plotly

import definitions.layout_styles as styles
from definitions.backend_calculations import detect_models, compute_overlap
from definitions.backend_dynamic_plots import plot_overlap

from definitions.ui_functions import single_result_ui, update_single_result, overlap_page

start_folder = './results'

# ======================================================================================================================

app_ui = ui.page_fillable(
    ui.page_navbar(
        ui.nav_spacer(),
        ui.nav_panel('Welcome to BrainMApp',
                     ui.markdown('</br>Welcome to **BrainMApp**!</br></br>This app will let you visualize your '
                                 'statistical surface maps in an interactive way.</br>To start, please input the path'
                                 ' to the directory where your project results are stored:'),
                     ui.div(ui.layout_columns(
                             ui.input_text(id="results_folder", label='', value=start_folder),
                             ui.input_action_button(id='go_button',
                                                    label='GO',
                                                    class_='btn btn-dark action-button'),
                             col_widths=(9, 3)
                     )),
                     ' ',  # Spacer
                     ui.output_ui(id='output_results_folder'),
                     ' ',
                     'Have fun!',
                     value='tab1'
                     ),
        ui.nav_panel('Main results',
                     ui.markdown('</br>Select the map(s) you want to see, with the settings you prefer and hit'
                                 ' **GO** to visualize the 3D brains.</br></br>Note: sometimes it can take a second to'
                                 ' draw those pretty brains, so you may need a little pinch of patience.'
                                 ' If you do not have that kind of time, you can reduce the resolution to low.</br>'),
                     single_result_ui('result1'),
                     single_result_ui('result2'),
                     ' ',  # Spacer
                     value='tab2'
                     ),
        ui.nav_panel('Overlap',
                     ' ',  # Spacer - fix with padding later or also never
                     overlap_page,
                     ' ',  # spacer
                     value='tab3'
                     ),
        title="BrainMApp: visualize your verywise output",
        selected='tab1',
        position='fixed-top',
        fillable=True,
        bg='white',
        window_title='BrainMApp',
        id='navbar'),

    padding=styles.PAGE_PADDING,
    gap=styles.PAGE_GAP,
)


def server(input, output, session):

    # TAB 2: MAIN RESULTS
    group1, model1, measure1 = update_single_result('result1', go=input.go_button, input_resdir=input.results_folder)
    group2, model2, measure2 = update_single_result('result2', go=input.go_button, input_resdir=input.results_folder)

    # TAB 1: FOLDER INFO
    @output
    @render.text
    @reactive.event(input.go_button)
    def output_results_folder():

        model_list = detect_models(input.results_folder())

        return ui.markdown(f'You have selected: {input.results_folder()}</br></br>'
                           f'This folder contains the following models: {model_list}</br></br>'
               f'Now, on the top right corner you can navigate to the **"Main results"** tab to choose which maps you'
               f' would like to see. If you select *two* maps on the Main results page you can then see their overlap'
               f' by navigating to the **"Overlap"** tab.')

    # TAB 3: OVERLAP
    @render.text
    def overlap_info():
        ovlp_info = compute_overlap(resdir=input.results_folder(),
                                    group1=group1(), model1=model1(), measure1=measure1(),
                                    group2=group2(), model2=model2(), measure2=measure2())[0]

        text = {}
        legend = {}
        for key in [1, 2, 3]:
            text[key] = f'**{ovlp_info[key][1]}%** ({ovlp_info[key][0]} vertices)' if key in ovlp_info.keys() else \
                '**0%** (0 vertices)'
            color = styles.OVLP_COLORS[key-1]
            legend[key] = f'<span style = "background-color: {color}; color: {color}"> oo</span>'

        return ui.markdown(f'There was a {text[3]} {legend[3]} **overlap** between the terms selected:</br>'
                           f'{text[1]} was unique to {legend[1]}  **{model1()}** (<ins>{measure1()}</ins>)</br>'
                           f'{text[2]} was unique to {legend[2]}  **{model2()}** (<ins>{measure2()}</ins>)')

    @reactive.Calc
    def overlap_brain3D():
        return plot_overlap(resdir=input.results_folder(),
                            group1=group1(), model1=model1(), measure1=measure1(),
                            group2=group2(), model2=model2(), measure2=measure2(),
                            surf=input.overlap_select_surface(),
                            resol=input.overlap_select_resolution())

    @render_plotly
    def overlap_brain_left():
        brain = overlap_brain3D()
        return brain['left']

    @render_plotly
    def overlap_brain_right():
        brain = overlap_brain3D()
        return brain['right']


app = App(app_ui, server)

