"""Test runner for the GUI Agent"""
import argparse
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import List
from urllib.parse import urlparse

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
from Mind2Web_evaluation.evaluator import LLMEvaluator
from utils.early_stop import early_stop
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

        evaluate_model = load_tool_llm(self.args)
        self.evaluator = LLMEvaluator(vllm_client=evaluate_model)

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

    def run(self, config_file_list: list[str]):
        """Run the main test loop"""
        
        # Process each config file
        for config_file in config_file_list:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            if 'google' in config_data['start_url']:
                continue
            self._process_config_file(config_file)
        
        # Close environment
        self.env.close()

        return self.metrics_tracker.save_summary()
    
    def _process_config_file(self, config_file: str):
        """Process a single config file"""
        sub_domain = Path(config_file).stem.rsplit('_', 1)[0]
        
        action_list = []
        score_value = None
        success_flag = False
        final_url = None
        task_identifier = None
        extra_metadata = {"config_file": config_file}

        render_helper = RenderHelper(config_file, self.args.result_dir)

        # Get intent and task info
        with open(config_file) as f:
            _c = json.load(f)
            intent = _c["intent"]  
            intent += ' Please yield the stop action once you have completed the task.'  
            task_id = _c["task_id"]
            site = urlparse(_c.get("start_url", "")).netloc
        
        episode_id = f"{task_id}"

        task_identifier = str(task_id)
        extra_metadata.update({
            "task_id": task_identifier,
            "sub_domain": sub_domain,
        })
        self.metrics_tracker.start_task(
            task_identifier,
            metadata={
                "intent": intent,
                "config_file": config_file,
                "sub_domain": sub_domain,
            },
        )

        try:
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
            if 'about:blank' in current_url or info["is_blocked"]:
                self.logger.info(f"[Result] (Cannot navigate to {_c['start_url']}) {config_file}")
                extra_metadata["failure_reason"] = "Navigation blocked"
                final_url = current_url
                return
            
            meta_data = {"action_history": [],
                         "response_history": [],
                         "site": site}
            
            print("config_file: ", config_file)

            all_view_url = []
            
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
                    all_view_url.append(current_url)

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

                    if isinstance(action, list):
                        last_action_type = action[-1]["action_type"]
                    else:
                        last_action_type = action["action_type"]
                    if last_action_type in [ActionTypes.STOP, 'finished']:
                        self.logger.info(f"[Sub-query {sub_query_idx + 1}] Completed")

                        # Store the subtask answer if using subtask decomposition
                        if self.args.subtask:
                            # Extract the answer from the stop action
                            if isinstance(action, list):
                                answer = action[-1].get('answer', '')
                            else:
                                answer = action.get('answer', '')

                            # Store the subtask intent and answer
                            sub_query_answers.append((enhanced_intent, answer))
                            self.logger.info(f"[Subtask {sub_query_idx + 1}] Answer stored: {answer[:100]}...")

                        break

                    try:
                        obs, _, terminated, _, info, current_url = self.env.step(action, observation=obs)
                    except Exception as e:
                        self.logger.error(f"Error in step: {e}")
                        extra_metadata["error"] = str(e)
                        final_url = current_url if isinstance(current_url, str) else None
                        return
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
                            self.logger.info(f"[Subtask {sub_query_idx + 1}] Terminated without completion")

                        break

            self.agent.experience_memory = None
            self.agent.experience_texts, self.agent.experience_images = None, None
            
            # evaluate the scores   
            score, answer_text, ori_answer = self.evaluator(config_file, self.args.result_dir)
            last_action = trajectory[-1]
            pred = last_action.get("answer", "")
            reasoning = last_action.get("reasoning", "")
            self.logger.info(f"[Result] Predicted answer: {pred}\nReasoning: {reasoning}")
        
            result = "PASS" if score==1 else "FAIL"
            self.logger.info(f"[Result] ({result}) {config_file}")
            
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
                        "task_description": intent
                    }
                    
                    # End the conversation
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
                tokens=None,
                final_url=final_url,
                score=score_value,
                extra_metadata=extra_metadata,
            )

            try:
                render_helper.close()
            except Exception:
                pass
    
