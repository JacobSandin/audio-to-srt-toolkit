#!/usr/bin/env python3
# Test runner script
# Runs all tests in the tests directory
# 2025-04-23 -JS

import os
import sys
import unittest
import pytest
import logging
import argparse

def log(level, *messages, **kwargs):
    """
    Unified logging function.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        messages: Messages to log
        kwargs: Additional logging parameters
    """
    logger = logging.getLogger(__name__)
    
    if level == logging.DEBUG:
        logger.debug(*messages, **kwargs)
    elif level == logging.INFO:
        logger.info(*messages, **kwargs)
    elif level == logging.WARNING:
        logger.warning(*messages, **kwargs)
    elif level == logging.ERROR:
        logger.error(*messages, **kwargs)
    elif level == logging.CRITICAL:
        logger.critical(*messages, **kwargs)

def setup_logging(debug=False, quiet=False):
    """
    Set up logging based on command-line arguments.
    
    Args:
        debug: Enable debug logging
        quiet: Suppress console output
    """
    # Determine log level
    if debug:
        console_level = logging.DEBUG
    elif quiet:
        console_level = logging.WARNING
    else:
        console_level = logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=console_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout) if not quiet or debug else logging.NullHandler()
        ]
    )

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run tests for Audio Toolkit')
    
    parser.add_argument(
        '--unittest',
        action='store_true',
        help='Run unittest tests'
    )
    
    parser.add_argument(
        '--pytest',
        action='store_true',
        help='Run pytest tests'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()

def run_unittest_tests():
    """
    Run tests using unittest.
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    log(logging.INFO, "Running unittest tests...")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
    suite = loader.discover(start_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_pytest_tests():
    """
    Run tests using pytest.
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    log(logging.INFO, "Running pytest tests...")
    
    # Get the tests directory
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
    
    # Run pytest
    result = pytest.main(['-v', tests_dir])
    
    return result == 0

def main():
    """
    Main entry point for the test runner.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.debug, args.quiet)
    
    try:
        log(logging.INFO, "Starting test runner")
        
        # Determine which tests to run
        run_unittest = args.unittest or not args.pytest
        run_pytest_flag = args.pytest or not args.unittest
        
        # Run tests
        success = True
        
        if run_unittest:
            unittest_success = run_unittest_tests()
            success = success and unittest_success
            
        if run_pytest_flag:
            pytest_success = run_pytest_tests()
            success = success and pytest_success
        
        # Report results
        if success:
            log(logging.INFO, "All tests passed")
            sys.exit(0)
        else:
            log(logging.ERROR, "Some tests failed")
            sys.exit(1)
            
    except Exception as e:
        log(logging.CRITICAL, f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
