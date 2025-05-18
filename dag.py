from graphviz import Digraph

import time
import random

class DAG:
      def __init__(self, pipelines, phases):
            """ Phases should be run through the simulation """

            self.pipelines = pipelines
            self.phases = phases #self._set_up_phases(phases)
            self.dot = Digraph(comment='ML Pipelines DAG', format='png')
            self.dot.attr(rankdir='TB', 
                  bgcolor='white',
                  labelloc='t',  # Move title to top
                  label=f'Machine Learning Pipeline Architecture\nRendered at {time.strftime("%Y-%m-%d %H:%M:%S")}',
                  fontsize='24',
                  fontname='Arial Bold',
                  nodesep='0.8',
                  ranksep='1.2',  # Increased for better spacing
                  pad='0.75',     # Increased padding
                  concentrate='true') # Merge edge lines where possible
            
      def _set_up_phases(self, phases):
            """ Set up the phases for the DAG """
            print("Setting thin up")
            phases_adjusted = {pipeline: {} for pipeline in self.pipelines}
            for pipeline in self.pipelines:
                  for phase in phases:
                        phases_adjusted[pipeline][phase] = {}
            return phases_adjusted
            
      def _get_random_color(self, category: str, index: int|None = None):
            if category == "pipeline":
                  # Stronger, more distinct pipeline colors
                  return "#5D6D7E"  # Dark slate blue
            elif category == "phase":
                  # More vibrant, distinct phase colors
                  possible_colors = [
                        "#2ECC71",  # Emerald green
                        "#3498DB",  # Bright blue
                        "#9B59B6",  # Amethyst purple
                        "#E74C3C",  # Bright red
                        "#1ABC9C",  # Turquoise
                        "#D35400",  # Pumpkin orange
                        "#34495E",  # Wet asphalt dark blue
                        "#E84393",  # Pink
                        "#8E44AD",  # Wisteria purple
                        "#16A085",  # Green sea
                  ]
                  return possible_colors[index % len(possible_colors)]
            elif category == "procedure":
                  # Distinct middle-tone colors
                  possible_colors = [
                        "#52BE80",  # Medium green
                        "#5DADE2",  # Medium blue
                        "#AF7AC5",  # Medium purple
                        "#EC7063",  # Medium red
                        "#45B39D",  # Medium turquoise
                        "#DC7633",  # Medium orange
                        "#5D6D7E",  # Medium slate
                        "#F1948A",  # Medium salmon
                        "#A569BD",  # Medium violet
                        "#76D7C4",  # Medium mint
                  ]
                  return possible_colors[index % len(possible_colors)]
            elif category == "subprocedure":
                  # Lighter but still distinct colors
                  possible_colors = [
                        "#ABEBC6",  # Light green
                        "#AED6F1",  # Light blue
                        "#D7BDE2",  # Light purple
                        "#F5B7B1",  # Light red
                        "#A3E4D7",  # Light turquoise
                        "#F5CBA7",  # Light orange
                        "#AEB6BF",  # Light slate
                        "#FADBD8",  # Light salmon
                        "#D2B4DE",  # Light violet
                        "#A2D9CE",  # Light mint
                  ]
                  return possible_colors[index % len(possible_colors)]
            elif category == "method":
                  # Very light but still differentiable colors
                  possible_colors = [
                        "#EAFAF1",  # Very light green
                        "#EBF5FB",  # Very light blue
                        "#F4ECF7",  # Very light purple
                        "#FDEDEC",  # Very light red
                        "#E8F8F5",  # Very light turquoise
                        "#FEF5E7",  # Very light orange
                        "#EBEDEF",  # Very light slate
                        "#FDEDEC",  # Very light salmon
                        "#F5EEF8",  # Very light violet
                        "#E8F6F3",  # Very light mint
                  ]
                  return possible_colors[index % len(possible_colors)]
    
      def _add_legend(self):
            """Add a legend subgraph to explain node types and colors"""
            with self.dot.subgraph(name='cluster_legend') as legend:
                  legend.attr(label='Legend (by brightness)', fontsize='18', fontname='Arial Bold', 
                        style='filled', fillcolor='#F8F9F9', penwidth='2')
                  
                  # Create a node for each category with its corresponding color
                  legend.node('legend_eda', 'EDA', style='filled', 
                        fillcolor='#FFB347', shape='ellipse', fontname='Arial')
                  legend.node('legend_pipeline', 'Pipeline', style='filled', 
                        fillcolor='#5D6D7E', shape='rect', fontname='Arial')
                  legend.node('legend_phase', 'Phase', style='filled', 
                        fillcolor='#2ECC71', shape='rect', fontname='Arial')
                  legend.node('legend_procedure', 'Procedure', style='filled', 
                        fillcolor='#52BE80', shape='rect', fontname='Arial')
                  legend.node('legend_subprocedure', 'Subprocedure', style='filled', 
                        fillcolor='#ABEBC6', shape='rect', fontname='Arial')
                  legend.node('legend_method', 'Method', style='filled', 
                        fillcolor='#EAFAF1', shape='rect', fontname='Arial')
                  
                  # Set invisible edges to arrange legend items vertically
                  legend.edge('legend_eda', 'legend_pipeline', style='invis')
                  legend.edge('legend_pipeline', 'legend_phase', style='invis')
                  legend.edge('legend_phase', 'legend_procedure', style='invis')
                  legend.edge('legend_procedure', 'legend_subprocedure', style='invis')
                  legend.edge('legend_subprocedure', 'legend_method', style='invis')
                  
                  # Force a left-to-right arrangement
                  legend.attr(rank='same')
                  legend.graph_attr.update(rankdir='LR')
      
      def _set_up_nodes(self):
            """ Set up the nodes for the DAG """
            self.dot.node('EDA', 'EDA', style='filled', fillcolor='#FFB347', shape='ellipse', 
                  fontname='Arial', fontsize='18', penwidth='2')

            for i, pipeline in enumerate(self.pipelines):
                  self.dot.node(pipeline, pipeline, style='filled', 
                        fillcolor=self._get_random_color("pipeline", i), 
                        shape='rect', fontname='Arial', fontsize='18', penwidth='1.5')
                  self.dot.edge('EDA', pipeline, penwidth='2')

            self.model_keys_to_pipeline_names = {
                  pipeline: [] for pipeline in self.pipelines
            }

            # Bind all the models to the pipelines 
            for i, (pipeline, models) in enumerate(self.pipelines.items()):

                  for modelName in models:
                        model_key = f"{pipeline}_{modelName}"
                        self.model_keys_to_pipeline_names[pipeline].append(model_key)
                        self.dot.node(model_key, modelName, style='filled', 
                              fillcolor=self._get_random_color("pipeline", i), 
                              shape='rect', fontname='Arial', fontsize='16')
                        self.dot.edge(pipeline, model_key, penwidth='1.5')
      
      def _format_node_label(self, main_text, comment=None):
            """Format node label with optional comment"""
            if comment:
                  return f"{main_text}\n({comment})"
            return main_text

      def _dag_draw_phases(self):
            """ Draw the phases for the DAG """
            for pipeline in self.pipelines:
                  prev_procedures_keys = []
                  for i, phase in enumerate(self.phases[pipeline]):
                        phase_node = f"{pipeline}_{phase}"
                        self.dot.node(phase_node,
                              phase,
                              style='filled',
                              shape='rect',
                              fontname='Arial',
                              fontsize='16',
                              penwidth='1.5',
                              fillcolor=self._get_random_color("phase", i)
                              )
                        if i == 0:
                              for model_key in self.model_keys_to_pipeline_names[pipeline]:
                                    self.dot.edge(model_key, phase_node, penwidth='1.5')

                        new_procedures_keys = []
                        for i, (procedure_name, procedure_data) in enumerate(self.phases[pipeline][phase].items()):
                              # Format procedure label with comment if it exists
                              comment = procedure_data.get('_comment')
                              label = self._format_node_label(procedure_name, comment)
                              shape = 'box' if comment else 'rect'

                              procedure_node = f"{phase_node}_{procedure_name}"
                              new_procedures_keys.append(procedure_node)
                              self.dot.node(procedure_node, label,
                                    style='filled',
                                    shape=shape,
                                    fontname='Arial',
                                    fillcolor=self._get_random_color("procedure", i),
                                    fontsize='14'
                                    )
                              self.dot.edge(phase_node, procedure_node)

                              # Handle subprocedures
                              for j, (subprocedure_name, subprocedure_data) in enumerate(procedure_data['_subprocedures'].items()):
                                    comment = subprocedure_data.get('_comment')
                                    label = self._format_node_label(subprocedure_name, comment)
                                    shape = 'box' if comment else 'rect'

                                    subprocedure_node = f"{procedure_node}_{subprocedure_name}"
                                    self.dot.node(subprocedure_node, label,
                                          style='filled',
                                          shape=shape,
                                          fontname='Arial',
                                          fillcolor=self._get_random_color("subprocedure", j),
                                          fontsize='12'
                                          )
                                    self.dot.edge(procedure_node, subprocedure_node)

                                    # Handle methods
                                    for k, (method_name, method_data) in enumerate(subprocedure_data['_methods'].items()):
                                          comment = method_data.get('_comment')
                                          label = self._format_node_label(method_name, comment)
                                          shape = 'box' if comment else 'rect'

                                          method_node = f"{subprocedure_node}_{method_name}"
                                          self.dot.node(method_node, label,
                                                style='filled',
                                                shape=shape,
                                                fontname='Arial',
                                                fillcolor=self._get_random_color("method", k),
                                                fontsize='10'
                                                )
                                          self.dot.edge(subprocedure_node, method_node)

                        if i > 0:
                              for prev_procedure in prev_procedures_keys:
                                    self.dot.edge(prev_procedure, phase_node)
                        prev_procedures_keys = new_procedures_keys
      
      def add_procedure(self, pipelineName, phaseName, procedureName, comment=None):
            """Add a procedure to a phase with an optional comment"""
            if procedureName not in self.phases[pipelineName][phaseName]:
                  self.phases[pipelineName][phaseName][procedureName] = {
                        '_comment': comment,  # Store comment separately
                        '_subprocedures': {}  # Store subprocedures in a separate dict
                  }

      def add_subprocedure(self, pipelineName, phaseName, procedureName, subprocedureName, comment=None):
            """Add a subprocedure to a procedure with an optional comment"""
            if procedureName in self.phases[pipelineName][phaseName]:
                  self.phases[pipelineName][phaseName][procedureName]['_subprocedures'][subprocedureName] = {
                        '_comment': comment,
                        '_methods': {}
                  }

      def add_method(self, pipelineName, phaseName, procedureName, subprocedureName, methodName, comment=None):
            """Add a method to a subprocedure with an optional comment"""
            if (procedureName in self.phases[pipelineName][phaseName] and 
                  subprocedureName in self.phases[pipelineName][phaseName][procedureName]['_subprocedures']):
                  self.phases[pipelineName][phaseName][procedureName]['_subprocedures'][subprocedureName]['_methods'][methodName] = {
                        '_comment': comment
                  }

      def render(self):
            self._set_up_nodes()
            self._dag_draw_phases()
            self._add_legend()
            output_path = self.dot.render(filename='ml_pipelines_dag')
            print(f'Graph saved to: {output_path}')

