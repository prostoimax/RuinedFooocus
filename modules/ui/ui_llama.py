import gradio as gr
from modules.llama_pipeline import run_llama, llama_names
from pathlib import Path

def add_llama_tab(prompt):
    def run_llama_run(system_file, prompt):
        res = run_llama(system_file, prompt)

        return gr.update(value=res)

    with gr.Group(), gr.Row():
        llama_btn = gr.Button(value="Run 🦙, run.")
        llama_select = gr.Dropdown(
            choices=llama_names(),
            label="Prompt rewrite Llama",
            show_label=True,
            buttons=[llama_btn],
        )
    llama_btn.click(
        run_llama_run,
        api_visibility='undocumented',
        inputs=[llama_select, prompt],
        outputs=[prompt]
    )
