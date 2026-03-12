"""Test runner for the GUI Agent"""
import argparse
import json
import logging
import os
import pickle
import re
import time
from pathlib import Path
import base64
import io
from typing import List

from browser_env import (
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from MMInA_evaluation.evaluator import evaluator_router
from utils.early_stop import early_stop
from utils.help_functions import is_domain_type, save_scores_to_json
from utils.metrics_tracker import InferenceMetricsTracker
from agent.llm_config import load_tool_llm


class TestRunner:
    """Handles the main test execution loop"""
    
    def __init__(self, args: argparse.Namespace, agent):
        self.args = args
        self.agent = agent
        self.logger = logging.getLogger("logger")

        # Build reliability config (opt-in)
        reliability_config = None
        if getattr(args, 'enable_reliability', False):
            from utils.action_retry import RetryConfig
            reliability_config = {
                'retry_config': RetryConfig(
                    max_retries=getattr(args, 'action_retry_max', 3),
                    base_delay=getattr(args, 'action_retry_base_delay', 0.5),
                ),
                'use_page_stability': getattr(args, 'use_page_stability', False),
                'stability_timeout': getattr(args, 'page_stability_timeout', 5000),
            }

        # Initialize environment
        self.env = ScriptBrowserEnv(
            headless=True,
            slow_mo=args.slow_mo,
            viewport_size={
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            save_trace_enabled=args.save_trace_enabled,
            sleep_after_execution=args.sleep_after_execution,
            args=args,  # Pass args to the environment
            reliability_config=reliability_config,
        )

        # Session monitor (opt-in)
        self.session_monitor = None
        if getattr(args, 'enable_reliability', False):
            from utils.session_monitor import SessionMonitor
            self.session_monitor = SessionMonitor({
                "degraded_threshold": getattr(args, 'session_degraded_threshold', 2),
                "critical_threshold": getattr(args, 'session_critical_threshold', 4),
                "failed_threshold": 6,
            })
        
        # Initialize token and timing tracking
        self.total_input_tokens = 0
        self.config_processing_times = {}
        self.config_token_counts = {}
        
        # Load existing token timing data if available
        self._load_existing_token_timing_data()

        run_metadata = {
            "evaluation_type": getattr(args, "evaluation_type", None),
            "domain": getattr(args, "domain", None),
            "model": getattr(args, "model", None),
            "use_memory": getattr(args, "use_memory", None),
            "use_continuous_memory": getattr(args, "use_continuous_memory", None),
            "memory_mode": getattr(args, "memory_mode", None),
            "enable_verifier": getattr(args, "enable_verifier", None),
        }
        self.metrics_tracker = InferenceMetricsTracker(args.result_dir, run_metadata)
    
    def _track_tokens(self, llm_wrapper):
        """Track input tokens from LLM wrapper's last usage info"""
        # try:
        prompt_tokens = llm_wrapper.last_usage['prompt_tokens']
        self.total_input_tokens += prompt_tokens
        return prompt_tokens
        # except Exception as e:
        #     self.logger.warning(f"Could not track tokens: {e}")
        # return 0
    
    def _load_existing_token_timing_data(self):
        """Load existing token timing data if available for resuming"""
        token_timing_path = Path(self.args.result_dir) / "token_timing_data.json"
        if token_timing_path.exists():
            try:
                with open(token_timing_path, 'r') as f:
                    existing_data = json.load(f)
                
                # Restore the tracking variables
                self.total_input_tokens = existing_data.get('total_input_tokens', 0)
                self.config_processing_times = existing_data.get('config_processing_times', {})
                self.config_token_counts = existing_data.get('config_token_counts', {})
                
                self.logger.info(f"[Resume] Loaded existing token timing data")
                self.logger.info(f"[Resume] Total tokens: {self.total_input_tokens}")
                self.logger.info(f"[Resume] Configs already processed: {len(self.config_processing_times)}")
                self.logger.info(f"[Resume] Total time so far: {sum(self.config_processing_times.values()):.2f} seconds")
                
            except Exception as e:
                self.logger.warning(f"Could not load existing token timing data: {e}")
                # Reset to defaults if loading fails
                self.total_input_tokens = 0
                self.config_processing_times = {}
                self.config_token_counts = {}
    
    def run(self, config_file_list: list[str]):
        """Run the main test loop"""
        
        # Process each config file
        for config_file in config_file_list:
            self._process_config_file(config_file)
        # Close environment
        self.env.close()

        return self.metrics_tracker.save_summary()
    
    def _process_config_file(self, config_file: str):
        """Process a single config file"""
        # Start timing for this config file
        start_time = time.time()
        config_tokens = 0
        score_value = None
        success_flag = False
        final_url = None
        task_identifier = None
        extra_metadata = {"config_file": config_file}
        
        sub_domain = config_file.replace('MMInA/','').split('/')[1]
        
        action_list = []

        render_helper = RenderHelper(config_file, self.args.result_dir)

        # Get intent and task info
        with open(config_file) as f:
            _c = json.load(f)
            intent = _c["intent"]    
            intent = intent.replace('https://library.kiwix.org/iewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing','https://www.wikipedia.org/')
            intent = intent.replace("http://localhost:7770/", "http://ec2-3-146-212-252.us-east-2.compute.amazonaws.com:7770/")
            if 'shopping' in intent:
                intent += "\n\n***NOTE: Please 1. directly search the product name 2.find the query product and click it 3. carefully check the product image and description to answer the question***"
            task_id = _c["task_id"]
            site = _c["sites"][0]

        task_identifier = str(task_id)
        extra_metadata.update({
            "task_id": task_identifier,
            "sub_domain": sub_domain,
            "site": site,
        })
        self.metrics_tracker.start_task(
            task_identifier,
            metadata={
                "intent": intent,
                "config_file": config_file,
                "sub_domain": sub_domain,
                "site": site,
            },
        )

        numbers = re.findall(r'\d+', config_file)
        self.args.task_cnt = int(numbers[0]) if numbers else None
        self.args.hop_cnt = 0
        
        self.logger.info(f"[Config file]: {config_file}")
        self.logger.info(f"[Intent]: {intent}")
        
        self.agent.reset(config_file)
        self.agent.current_step = 0
        trajectory: Trajectory = []

        # Reset session monitor for new task
        if self.session_monitor:
            self.session_monitor.reset()
            self.agent.set_session_monitor(self.session_monitor)
        
        # Environment reset
        obs, info = self.env.reset(
            options={"config_file": config_file}, 
        )
        current_url = info["page"].url
        state_info: StateInfo = {"observation": obs, "info": info, "current_url": current_url}
        trajectory.append(state_info)
        print("CURRENT: ", current_url)
        
        meta_data = {"action_history": [],
                     "response_history": [],
                     "site": site}
        
        print("config_file: ", config_file)
        
        # Information accumulation storage
        sub_query_answers = []
        
        # Start conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            from utils.training_data_collector import get_collector
            collector = get_collector()
            if collector and collector.enabled:
                # Create conversation ID from task info
                conversation_id = f"{sub_domain}_{config_file.split('/')[-1].split('.')[0]}"
                collector.start_conversation(
                    conversation_id=conversation_id,
                    task_description=intent
                )
                self.logger.info(f"Started conversation collection for task: {conversation_id}")
        
        if self.args.subtask:
            # Implement subtask decomposition with LLM
            intent_list = self._decompose_task_into_subtasks(intent)
            self.logger.info(f"Task decomposed into {len(intent_list)} subtasks")
        else:
            intent_list = [intent]

        try:
            # Process each sub-query sequentially
            for sub_query_idx, current_intent in enumerate(intent_list):
                # Enhance current intent with previous subtask results if available
                if self.args.subtask and sub_query_idx > 0 and sub_query_answers:
                    enhanced_intent = self._enhance_intent_with_previous_results(
                        current_intent, sub_query_answers, sub_query_idx
                    )
                    self.logger.info(f"[Subtask {sub_query_idx + 1}] Enhanced intent with previous results")
                else:
                    enhanced_intent = current_intent

                # Reset environment for each sub-query if not the first one
                if sub_query_idx > 0:
                    self.logger.info(f"[Sub-query {sub_query_idx + 1}] Resetting environment for new sub-query")
                    obs, info = self.env.reset(options={"config_file": config_file})
                    current_url = info["page"].url
                    state_info: StateInfo = {"observation": obs, "info": info, "current_url": current_url}
                    # Clear trajectory and start fresh for new sub-query
                    trajectory = [state_info]
                    meta_data = {"action_history": [],
                                 "response_history": [],
                                 "page": self.env.page,
                                 "site": site}
                    print("CURRENT: ", current_url)

                # Process current sub-query
                while True:
                    current_url = current_url.lower()

                    early_stop_flag, stop_info = early_stop(
                        trajectory, self.args.max_steps, {
                            "parsing_failure": self.args.parsing_failure_th,
                            "repeating_action": self.args.repeating_action_failure_th,
                        },
                        session_monitor=self.session_monitor,
                    )

                    if early_stop_flag:
                        action = create_stop_action(f"Early stop: {stop_info}")
                    else:
                        def gen_action(intent, meta):
                            action, meta =  self.agent.next_action_custom(
                                trajectory,
                                intent,
                                meta_data=meta,
                            )
                            return action, meta
                        
                        action, meta_data = gen_action(enhanced_intent, meta_data)
                    
                    if isinstance(action, list):
                        trajectory.extend(action)
                    else:
                        trajectory.append(action)
                    
                    action_str = get_action_description(action)
                    render_helper.render(
                        action, state_info, meta_data, self.args.render_screenshot
                    )
                    meta_data["action_history"].append(action_str)
                    meta_data["page"] = self.env.page
                    
                    # Draw action and intent on the image for debug or memory saving
                    from PIL import ImageDraw, ImageFont, Image

                    image_bytes = base64.b64decode(obs["image"])
                    rendered_im = Image.open(io.BytesIO(image_bytes))
                    draw = ImageDraw.Draw(rendered_im)
                    # Example fonts; adjust if needed
                    font_large = ImageFont.load_default()
                    font_small = ImageFont.load_default()

                    draw.text((40, 40), action_str, fill=(0, 0, 0), font=font_large)
                    draw.text((40, 80), current_intent, fill=(0, 0, 0), font=font_small)
                    # Save the rendered image
                    model_res_path = os.path.join(self.args.result_dir, self.args.model, self.args.domain)
                    if self.args.hist:
                        model_res_path = os.path.join(model_res_path, f'hist_{self.args.hist_num}')
                    task_res_path = os.path.join(model_res_path, f"task_{self.args.task_cnt}")
                    hop_res_path = os.path.join(task_res_path, f"hop_{max(self.args.hop_cnt-1, 0)}")
                    image_dir = os.path.join(hop_res_path, "images")
                    # os.makedirs(image_dir, exist_ok=True)
                    # rendered_im.save(os.path.join(image_dir, f"{count}.png"))

                    if isinstance(action, list):
                        last_action_type = action[-1]["action_type"]
                    else:
                        last_action_type = action["action_type"]
                    if last_action_type in [ActionTypes.STOP, 'finished']:
                        self.logger.info(f"[Sub-query {sub_query_idx + 1}] Completed")
                        
                        # Store the subtask answer if using subtask decomposition
                        if self.args.subtask:
                            if isinstance(action, list):
                                answer = action[-1].get('answer', '')
                            else:
                                answer = action.get('answer', '')
                            sub_query_answers.append((enhanced_intent, answer))
                            self.logger.info(
                                f"[Subtask {sub_query_idx + 1}] Answer stored: {answer[:100]}..."
                            )
                        
                        break
                    
                    obs, _, terminated, _, info, current_url = self.env.step(action, observation=obs)
                    # observation, 0.0, done, truncated, info
                    print("CURRENT: ", current_url)

                    # Record step result in session monitor
                    if self.session_monitor:
                        action_error = info.get('action_error')
                        health = self.session_monitor.record_step(action_error, current_url)
                        self.logger.info(
                            f"[Session] Health: {health.value}"
                            + (f" Error: {action_error.category.value}" if action_error else "")
                        )

                    state_info = {"observation": obs, "info": info}
                    trajectory.append(state_info)

                    if terminated:
                        # add a action place holder
                        trajectory.append(create_stop_action(""))
                        self.logger.info(f"[Sub-query {sub_query_idx + 1}] Terminated")
                        
                        # Store the subtask answer if using subtask decomposition
                        if self.args.subtask:
                            sub_query_answers.append((enhanced_intent, "Task terminated without completion"))
                            self.logger.info(
                                f"[Subtask {sub_query_idx + 1}] Terminated without completion"
                            )
                        
                        break
                
            # evaluate the scores
            evaluate_model = load_tool_llm(self.args)
            evaluator = evaluator_router(config_file, evaluate_model)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=self.env.page,
                client=self.env.get_page_client(self.env.page),
            )
            last_action = trajectory[-1]
            pred = last_action.get("answer", "")
            reasoning = last_action.get("reasoning", "")
            score = 0.0 if 'Early stop' in pred else score
            self.logger.info(f"[Result] Predicted answer: {pred}\nReasoning: {reasoning}")
            result = "PASS" if score == 1 else "FAIL"
            self.logger.info(f"[Result] ({result}) {config_file}")

            self.agent.experience_memory = None
            self.agent.experience_texts, self.agent.experience_images = None, None

            # Calculate and log processing time and token usage for this config file
            end_time = time.time()
            processing_time = end_time - start_time

            # Store timing and token data
            self.config_processing_times[config_file] = processing_time
            self.config_token_counts[config_file] = config_tokens

            # Log only essential agent results (no token/timing details)
            self.logger.info(f"[Result] {config_file} - Success: {score}")

            # Save token and timing data incrementally after each config file
            self._save_token_timing_data_incremental()

            # End conversation for this task if training data collection is enabled
            if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
                from utils.training_data_collector import get_collector
                collector = get_collector()
                if collector and collector.enabled and collector.current_conversation_id:
                    # Create conversation summary
                    conversation_summary = {
                        "task_id": config_file.split('/')[-1].split('.')[0],
                        "site": site,
                        "sub_domain": sub_domain,
                        "domain": self.args.domain,
                        "evaluation_type": self.args.evaluation_type,
                        "model": self.args.model,
                        "success": score,
                        "final_url": current_url,
                        "task_completed": True,
                        "task_description": intent,
                    }

                    if self.args.collect_training_data:
                        saved_file = collector.end_conversation(conversation_summary, score)
                        if saved_file:
                            self.logger.info(f"Conversation saved: {saved_file}")

            score_value = score
            success_flag = result == "PASS"
            final_url = current_url

        except Exception as exc:
            extra_metadata["exception"] = str(exc)
            raise
        finally:
            tokens_value = config_tokens if config_tokens else None
            action_history = meta_data.get("action_history", []) if 'meta_data' in locals() else []
            extra_metadata.update({
                "repeated_action_count": sum(
                    1 for idx in range(1, len(action_history))
                    if action_history[idx] == action_history[idx - 1]
                ),
                "verifier_interventions": meta_data.get("verifier_interventions", 0) if 'meta_data' in locals() else 0,
                "memory_refreshes": meta_data.get("memory_refreshes", 0) if 'meta_data' in locals() else 0,
                "memory_hits": meta_data.get("memory_hits", 0) if 'meta_data' in locals() else 0,
            })
            self.metrics_tracker.end_task(
                task_identifier or config_file,
                success=success_flag,
                steps=getattr(self.agent, "current_step", None),
                tokens=tokens_value,
                final_url=final_url,
                score=score_value,
                extra_metadata=extra_metadata,
            )

            try:
                render_helper.close()
            except Exception:
                pass
    
    def _save_token_timing_data_incremental(self):
        """Save token and timing data incrementally after each config file"""
        # Calculate comprehensive statistics
        total_configs = len(self.config_processing_times)
        total_time = sum(self.config_processing_times.values())
        total_tokens = sum(self.config_token_counts.values())
        
        if total_configs > 0:
            avg_time = total_time / total_configs
            avg_tokens = total_tokens / total_configs
            min_time = min(self.config_processing_times.values())
            max_time = max(self.config_processing_times.values())
            min_tokens = min(self.config_token_counts.values())
            max_tokens = max(self.config_token_counts.values())
        else:
            avg_time = avg_tokens = min_time = max_time = min_tokens = max_tokens = 0
        
        token_timing_data = {
            "total_input_tokens": self.total_input_tokens,
            "config_processing_times": self.config_processing_times,
            "config_token_counts": self.config_token_counts,
            "summary": {
                "total_configs_processed": total_configs,
                "total_processing_time": total_time,
                "average_processing_time": avg_time,
                "min_processing_time": min_time,
                "max_processing_time": max_time,
                "total_tokens": total_tokens,
                "average_tokens_per_config": avg_tokens,
                "min_tokens_per_config": min_tokens,
                "max_tokens_per_config": max_tokens
            }
        }
        
        with open(Path(self.args.result_dir) / "token_timing_data.json", "w") as f:
            json.dump(token_timing_data, f, indent=4)
        
        # Save token and timing data to JSON (no logger output)
    
