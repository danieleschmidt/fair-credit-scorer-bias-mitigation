#!/usr/bin/env python3
"""
Autonomous Backlog Executor

This is the main entry point for the autonomous backlog management system.
It implements the continuous execution loop that:

1. Syncs & refreshes backlog from all sources
2. Discovers new tasks automatically 
3. Scores and ranks all items using WSJF methodology
4. Executes ready items using TDD micro-cycles
5. Maintains quality gates and security checks
6. Reports metrics and status continuously
7. Loops until all actionable work is complete

Usage:
    python autonomous_backlog_executor.py [--dry-run] [--max-cycles N] [--cycle-delay SECONDS]
"""

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from backlog_manager import BacklogManager, BacklogItem, TaskStatus
from logging_config import setup_logging


class AutonomousBacklogExecutor:
    """Main autonomous backlog execution engine"""
    
    def __init__(self, repo_path: str = "/root/repo", dry_run: bool = False):
        self.repo_path = repo_path
        self.dry_run = dry_run
        self.running = True
        self.cycle_count = 0
        self.manager = BacklogManager(repo_path)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger = logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def execute_full_backlog_cycle(self, max_cycles: Optional[int] = None, cycle_delay: float = 5.0) -> dict:
        """
        Execute the full autonomous backlog management cycle.
        
        Args:
            max_cycles: Maximum number of cycles to run (None for unlimited)
            cycle_delay: Delay between cycles in seconds
            
        Returns:
            Final execution statistics
        """
        
        start_time = time.time()
        total_completed = 0
        total_discovered = 0
        errors = []
        
        self.logger.info("🚀 Starting Autonomous Backlog Management System")
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        self.logger.info(f"Repository: {self.repo_path}")
        
        try:
            while self.running:
                cycle_start = time.time()
                self.cycle_count += 1
                
                self.logger.info(f"📋 Starting Cycle #{self.cycle_count}")
                
                # Check max cycles limit
                if max_cycles and self.cycle_count > max_cycles:
                    self.logger.info(f"Reached maximum cycles limit ({max_cycles})")
                    break
                
                try:
                    # === PHASE 1: SYNC & REFRESH ===
                    self.logger.info("🔄 Phase 1: Sync & Refresh")
                    self.manager.load_backlog()
                    initial_size = len(self.manager.backlog)
                    
                    # === PHASE 2: DISCOVER NEW TASKS ===
                    self.logger.info("🔍 Phase 2: Task Discovery")
                    new_tasks = self.manager.discover_new_tasks()
                    
                    if new_tasks:
                        self.logger.info(f"📝 Discovered {len(new_tasks)} new tasks")
                        for task in new_tasks:
                            self.logger.info(f"  - {task.task_type.value}: {task.title}")
                        
                        self.manager.backlog.extend(new_tasks)
                        total_discovered += len(new_tasks)
                    else:
                        self.logger.info("✓ No new tasks discovered")
                    
                    # === PHASE 3: SCORE & RANK ===
                    self.logger.info("📊 Phase 3: Score & Rank")
                    self.manager.score_and_rank()
                    
                    # Log top priorities
                    top_items = self.manager.backlog[:5]
                    self.logger.info("🏆 Top Priority Items:")
                    for i, item in enumerate(top_items, 1):
                        status_emoji = self._get_status_emoji(item.status)
                        self.logger.info(f"  {i}. {status_emoji} {item.title} (Score: {item.final_score:.2f})")
                    
                    # === PHASE 4: EXECUTE READY ITEMS ===
                    self.logger.info("⚡ Phase 4: Execute Ready Items")
                    
                    executed_this_cycle = 0
                    max_executions_per_cycle = 3  # Prevent infinite loops
                    
                    while executed_this_cycle < max_executions_per_cycle:
                        next_item = self.manager.get_next_ready_item()
                        
                        if not next_item:
                            self.logger.info("✓ No more ready items to execute")
                            break
                        
                        self.logger.info(f"🎯 Executing: {next_item.title}")
                        
                        if self.dry_run:
                            self.logger.info("🔍 DRY RUN: Simulating execution...")
                            next_item.status = TaskStatus.PR  # Simulate success
                            success = True
                        else:
                            success = self.manager.execute_item_tdd_cycle(next_item)
                        
                        if success:
                            self.logger.info(f"✅ Successfully executed: {next_item.title}")
                            total_completed += 1
                            executed_this_cycle += 1
                            
                            # Mark as merged/done for completed items
                            if next_item.status == TaskStatus.PR:
                                next_item.status = TaskStatus.MERGED
                                next_item.last_updated = time.time()
                        else:
                            self.logger.warning(f"❌ Failed to execute: {next_item.title}")
                            if next_item.blocked_reason:
                                self.logger.warning(f"   Reason: {next_item.blocked_reason}")
                            break  # Don't try more items if one fails
                    
                    # === PHASE 5: QUALITY GATES ===
                    self.logger.info("🔐 Phase 5: Quality Gates")
                    
                    if not self.dry_run:
                        # Run security scan
                        security_issues = self.manager.discovery_engine.scan_security_vulnerabilities()
                        if security_issues:
                            self.logger.warning(f"⚠️  Found {len(security_issues)} security issues")
                            # Convert to backlog items
                            for issue in security_issues:
                                security_item = BacklogItem(
                                    id=f"security_{int(time.time())}_{len(self.manager.backlog)}",
                                    title=f"Security: {issue.content[:50]}...",
                                    description=issue.content,
                                    task_type=issue.task_type,
                                    business_value=13,  # Max priority for security
                                    effort=5,
                                    status=TaskStatus.NEW
                                )
                                self.manager.backlog.append(security_item)
                    
                    # === PHASE 6: SAVE STATE & REPORT ===
                    self.logger.info("💾 Phase 6: Save State & Report")
                    
                    self.manager.save_backlog()
                    report = self.manager.generate_status_report()
                    
                    # Log key metrics
                    self.logger.info("📈 Cycle Summary:")
                    self.logger.info(f"  • Backlog Size: {report['backlog_size']}")
                    self.logger.info(f"  • Items Executed: {executed_this_cycle}")
                    self.logger.info(f"  • Blocked Items: {len(report['blocked_items'])}")
                    self.logger.info(f"  • Avg WSJF Score: {report['avg_wsjf_score']:.2f}")
                    
                    cycle_duration = time.time() - cycle_start
                    self.logger.info(f"⏱️  Cycle Duration: {cycle_duration:.2f}s")
                    
                except Exception as e:
                    self.logger.error(f"💥 Cycle {self.cycle_count} failed: {e}")
                    errors.append(f"Cycle {self.cycle_count}: {str(e)}")
                
                # === PHASE 7: CHECK COMPLETION ===
                ready_items = [item for item in self.manager.backlog 
                             if item.is_ready() and not item.is_blocked()]
                
                if not ready_items:
                    self.logger.info("🎉 All actionable items completed!")
                    break
                
                # Sleep between cycles
                if self.running and cycle_delay > 0:
                    self.logger.info(f"😴 Sleeping {cycle_delay}s before next cycle...")
                    time.sleep(cycle_delay)
        
        except KeyboardInterrupt:
            self.logger.info("👋 Execution interrupted by user")
        except Exception as e:
            self.logger.error(f"💥 Fatal error: {e}")
            errors.append(f"Fatal: {str(e)}")
        
        # === FINAL REPORT ===
        total_duration = time.time() - start_time
        
        final_report = {
            'cycles_completed': self.cycle_count,
            'total_duration': total_duration,
            'items_completed': total_completed,
            'items_discovered': total_discovered,
            'errors': errors,
            'final_backlog_size': len(self.manager.backlog),
            'items_per_minute': (total_completed / (total_duration / 60)) if total_duration > 0 else 0
        }
        
        self.logger.info("📊 FINAL EXECUTION REPORT")
        self.logger.info("=" * 50)
        self.logger.info(f"🔄 Cycles Completed: {final_report['cycles_completed']}")
        self.logger.info(f"⏰ Total Duration: {total_duration:.2f}s ({total_duration/60:.1f}m)")
        self.logger.info(f"✅ Items Completed: {final_report['items_completed']}")
        self.logger.info(f"🔍 Items Discovered: {final_report['items_discovered']}")
        self.logger.info(f"📋 Final Backlog Size: {final_report['final_backlog_size']}")
        self.logger.info(f"⚡ Completion Rate: {final_report['items_per_minute']:.2f} items/min")
        
        if errors:
            self.logger.warning(f"⚠️  Errors Encountered: {len(errors)}")
            for error in errors[-5:]:  # Show last 5 errors
                self.logger.warning(f"   - {error}")
        
        return final_report
    
    def _get_status_emoji(self, status: TaskStatus) -> str:
        """Get emoji representation for task status"""
        emoji_map = {
            TaskStatus.NEW: "🆕",
            TaskStatus.REFINED: "📝", 
            TaskStatus.READY: "🚀",
            TaskStatus.DOING: "⚡",
            TaskStatus.PR: "🔄",
            TaskStatus.BLOCKED: "🚫",
            TaskStatus.MERGED: "🎯",
            TaskStatus.DONE: "✅"
        }
        return emoji_map.get(status, "❓")
    
    def get_backlog_summary(self) -> dict:
        """Get current backlog summary without executing"""
        self.manager.load_backlog()
        self.manager.score_and_rank()
        
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(1 for item in self.manager.backlog if item.status == status)
        
        return {
            'total_items': len(self.manager.backlog),
            'status_distribution': status_counts,
            'top_items': [
                {
                    'id': item.id,
                    'title': item.title,
                    'type': item.task_type.value,
                    'score': item.final_score,
                    'status': item.status.value
                }
                for item in self.manager.backlog[:10]
            ],
            'ready_items': len([item for item in self.manager.backlog if item.is_ready()]),
            'blocked_items': len([item for item in self.manager.backlog if item.is_blocked()])
        }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Backlog Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run continuous backlog management
  python autonomous_backlog_executor.py

  # Dry run to see what would happen
  python autonomous_backlog_executor.py --dry-run

  # Run for maximum 5 cycles with 10s delays
  python autonomous_backlog_executor.py --max-cycles 5 --cycle-delay 10

  # Just show current backlog status
  python autonomous_backlog_executor.py --status-only
        """
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Simulate execution without making changes"
    )
    
    parser.add_argument(
        "--max-cycles",
        type=int,
        help="Maximum number of execution cycles (default: unlimited)"
    )
    
    parser.add_argument(
        "--cycle-delay",
        type=float,
        default=5.0,
        help="Delay between cycles in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "--repo-path",
        default="/root/repo",
        help="Path to repository (default: /root/repo)"
    )
    
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Show backlog status and exit"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(config_override={'default_level': args.log_level})
    logger = logging.getLogger(__name__)
    
    try:
        executor = AutonomousBacklogExecutor(
            repo_path=args.repo_path,
            dry_run=args.dry_run
        )
        
        if args.status_only:
            # Just show status and exit
            summary = executor.get_backlog_summary()
            
            print("\n📋 BACKLOG STATUS SUMMARY")
            print("=" * 40)
            print(f"Total Items: {summary['total_items']}")
            print(f"Ready Items: {summary['ready_items']}")
            print(f"Blocked Items: {summary['blocked_items']}")
            
            print("\n📊 Status Distribution:")
            for status, count in summary['status_distribution'].items():
                if count > 0:
                    print(f"  {status}: {count}")
            
            print("\n🏆 Top Priority Items:")
            for i, item in enumerate(summary['top_items'][:5], 1):
                print(f"  {i}. [{item['status']}] {item['title']} (Score: {item['score']:.2f})")
                
            return
        
        # Run full execution
        final_report = executor.execute_full_backlog_cycle(
            max_cycles=args.max_cycles,
            cycle_delay=args.cycle_delay
        )
        
        # Exit with appropriate code
        if final_report['errors']:
            logger.warning("Execution completed with errors")
            sys.exit(1)
        else:
            logger.info("Execution completed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Failed to start autonomous backlog executor: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()