from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from shinywidgets import output_widget, render_plotly

import definitions.layout_styles as styles
from definitions.backend_calculations import detect_models, extract_results, compute_overlap
from definitions.backend_dynamic_plots import plot_surfmap, plot_overlap
from definitions.backend_static_plots import beta_colorbar_density_figure, clusterwise_means_figure


@module.ui
def single_result_ui():

    pheno_choice = ui.output_ui('pheno_ui')

    model_choice = ui.output_ui('model_ui')

    measure_choice = ui.input_selectize(
        id='select_measure',
        label='Measure',
        choices={'thickness': 'Thickness', 'area': 'Surface area'},
        selected='betas')

    output_choice = ui.input_selectize(
        id='select_output',
        label='Display',
        choices={'betas': 'Beta values', 'clusters': 'Clusters'},
        selected='betas')

    surface_choice = ui.input_selectize(
        id='select_surface',
        label='Surface type',
        choices={'pial': 'Pial', 'infl': 'Inflated', 'flat': 'Flat'},
        selected='pial')

    resolution_choice = ui.input_selectize(
        id='select_resolution',
        label='Resolution',
        choices={'fsaverage': 'High (164k nodes)', 'fsaverage6': 'Medium (50k nodes)', 'fsaverage5': 'Low (10k modes)'},
        selected='fsaverage6')

    # Buttons
    update_button = ui.div(ui.input_action_button(id='update_button',
                                                  label='GO',
                                                  class_='btn btn-dark action-button'),
                           style='padding-top: 15px')

    download_figure_button = ui.div(ui.input_action_button(id='download_figure_button',
                                                           label='Download png',
                                                           class_='btn btn-light action-button'),
                                    style='padding-top: 15px')

    return ui.div(
        # Selection pane
        ui.layout_columns(
            ui.layout_columns(
                pheno_choice, model_choice, measure_choice, output_choice, surface_choice, resolution_choice,
                col_widths=(2, 2, 2, 2, 2, 2),  # negative numbers for empty spaces
                gap='30px',
                style=styles.SELECTION_PANE),
            update_button,
            col_widths=(11, 1)
        ),
        # Info
        ui.layout_columns(
            ui.row(ui.output_ui('info'), style=styles.INFO_MESSAGE),
            download_figure_button,
            col_widths=(8, -2, 2)
        ),
        # Brain plots
        ui.layout_columns(
            ui.card('Left hemisphere',
                    output_widget('brain_left'),
                    full_screen=True),  # expand icon appears when hovering over the card body
            ui.card('Right hemisphere',
                    output_widget('brain_right'),
                    full_screen=True),
            ui.output_plot('colorbar_beta_histogram'),
            col_widths=(4, 4, 4)
        ))

@module.server
def update_single_result(input: Inputs, output: Outputs, session: Session,
                         go, input_resdir) -> tuple:

    # resdir = reactive.value(input_resdir)

    @output

    @render.ui
    @reactive.event(go)
    def pheno_ui():
        phenotypes = list(detect_models(input_resdir()).keys())
        return ui.input_selectize(
            id='select_pheno',
            label="Choose phenotype",
            choices=phenotypes,
            selected=phenotypes[0])

    @render.ui
    def model_ui():
        pheno = input.select_pheno()
        models = detect_models(input_resdir())[pheno]
        return ui.input_selectize(
            id='select_model',
            label='Choose model',
            choices=models,
            selected=models[0])  # start_model

    @reactive.Calc
    @reactive.event(input.update_button, ignore_none=False)
    def single_result_output():

        # Extract results
        min_beta, max_beta, mean_beta, n_clusters, sign_clusters, sign_betas, all_betas = extract_results(
            resdir=input_resdir(),
            group=input.select_pheno(),
            model=input.select_model(),
            measure=input.select_measure())

        l_nc = int(n_clusters[0])
        r_nc = int(n_clusters[1])

        if l_nc == r_nc == 0:
            info = ui.markdown(
                f'**0** clusters identified (in the left or the right hemisphere).')
            brains = {'left': None, 'right': None}
            legend_plot = None

        else:
            info = ui.markdown(
                f'**{l_nc + r_nc}** clusters identified ({l_nc} in the left and {r_nc} in the right hemisphere).<br />'
                f'Mean beta value [range] = **{mean_beta:.2f}** [{min_beta:.2f}; {max_beta:.2f}]')

            brains = plot_surfmap(
                min_beta, max_beta, n_clusters, sign_clusters, sign_betas,
                surf=input.select_surface(),
                resol=input.select_resolution(),
                output=input.select_output())

            if input.select_output() == 'betas':
                legend_plot = beta_colorbar_density_figure(sign_betas, all_betas,
                                                         figsize=(4, 6),
                                                         colorblind=False,
                                                         set_range=None)
            else:
                legend_plot = clusterwise_means_figure(sign_clusters, sign_betas,
                                                       figsize=(4, 6),
                                                       cmap=styles.CLUSTER_COLORMAP,
                                                       tot_clusters=int(n_clusters[0]+n_clusters[1]))

        return info, brains, legend_plot

    @render.text
    def info():
        md_info = single_result_output()[0]
        return md_info

    @render_plotly
    def brain_left():
        brain = single_result_output()[1]
        return brain['left']

    @render_plotly
    def brain_right():
        brain = single_result_output()[1]
        return brain['right']

    @render.plot(alt="All observed beta values")
    def colorbar_beta_histogram():
        return single_result_output()[2]

    return input.select_pheno, input.select_model, input.select_measure

# ------------------------------------------------------------------------------


overlap_page = ui.div(
        # Selection pane
        ui.layout_columns(
            ui.input_selectize(
                id='overlap_select_surface',
                label='Surface type',
                choices={'pial': 'Pial', 'infl': 'Inflated', 'flat': 'Flat'},
                selected='pial'),
            ui.input_selectize(
                id='overlap_select_resolution',
                label='Resolution',
                choices={'fsaverage': 'High (164k nodes)', 'fsaverage6': 'Medium (50k nodes)', 'fsaverage5': 'Low (10k modes)'},
                selected='fsaverage6'),

            ui.div(' ', style='padding-top: 80px'),

            col_widths=(3, 3, 2),  # negative numbers for empty spaces
            gap='30px',
            style=styles.SELECTION_PANE
        ),
        # Info
        ui.row(
            ui.output_ui('overlap_info'),
            style=styles.INFO_MESSAGE
        ),
        # Brain plots
        ui.layout_columns(
            ui.card('Left hemisphere',
                    output_widget('overlap_brain_left'),
                    full_screen=True),  # expand icon appears when hovering over the card body
            ui.card('Right hemisphere',
                    output_widget('overlap_brain_right'),
                    full_screen=True)
        ))

