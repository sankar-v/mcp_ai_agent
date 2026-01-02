"""
Test script for the MCP AI Agent workflow endpoint
Demonstrates the complete working of the agent with various queries
"""

import httpx
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


async def test_workflow(message: str, description: str):
    """Test the workflow endpoint with a message"""
    console.print(f"\n[bold cyan]Test: {description}[/bold cyan]")
    console.print(f"[yellow]Query:[/yellow] {message}\n")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8000/workflow",
                params={"message": message}
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Display response
            console.print(Panel(
                Markdown(str(result)),
                title="[bold green]Agent Response[/bold green]",
                border_style="green"
            ))
            
            return True
            
    except httpx.ConnectError:
        console.print("[bold red]Error: Cannot connect to the server.[/bold red]")
        console.print("Make sure the server is running: [cyan]uvicorn main:app --reload[/cyan]")
        return False
    except httpx.TimeoutException:
        console.print("[bold red]Error: Request timed out.[/bold red]")
        return False
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return False


async def run_tests():
    """Run a series of test queries"""
    console.print(Panel.fit(
        "[bold magenta]MCP AI Agent Test Suite[/bold magenta]\n"
        "Testing the workflow endpoint with various queries",
        border_style="magenta"
    ))
    
    tests = [
        {
            "message": "What is the latest news about artificial intelligence?",
            "description": "Testing Wikipedia tool with AI news query"
        },
        {
            "message": "Tell me about France",
            "description": "Testing country details tool"
        },
        {
            "message": "What are the details about Japan?",
            "description": "Testing country details with another country"
        },
        {
            "message": "Give me news about Python programming language",
            "description": "Testing Wikipedia tool with technology query"
        },
        {
            "message": "What is the capital of Brazil and give me its details?",
            "description": "Testing agent's ability to use both tools"
        }
    ]
    
    successful_tests = 0
    
    for test in tests:
        success = await test_workflow(test["message"], test["description"])
        if success:
            successful_tests += 1
        else:
            # If connection fails, stop testing
            break
        
        # Wait a bit between tests
        await asyncio.sleep(1)
    
    # Summary
    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print(f"[bold cyan]Test Summary:[/bold cyan]")
    console.print(f"Completed: [green]{successful_tests}[/green] out of [cyan]{len(tests)}[/cyan] tests")
    console.print(f"[bold]{'='*60}[/bold]\n")


async def interactive_mode():
    """Interactive mode to test custom queries"""
    console.print(Panel.fit(
        "[bold magenta]Interactive Mode[/bold magenta]\n"
        "Type your questions and see the agent respond!\n"
        "Type 'exit' or 'quit' to stop.",
        border_style="magenta"
    ))
    
    while True:
        console.print("\n[bold yellow]Your question:[/bold yellow] ", end="")
        message = input().strip()
        
        if message.lower() in ['exit', 'quit', 'q']:
            console.print("[cyan]Goodbye![/cyan]")
            break
        
        if not message:
            continue
        
        await test_workflow(message, "Custom Query")


def main():
    """Main entry point"""
    import sys
    
    console.print("\n[bold green]MCP AI Agent Test Program[/bold green]\n")
    console.print("Choose a mode:")
    console.print("  1. Run automated test suite")
    console.print("  2. Interactive mode (ask your own questions)")
    console.print("  3. Exit\n")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(run_tests())
    elif choice == "2":
        asyncio.run(interactive_mode())
    elif choice == "3":
        console.print("[cyan]Goodbye![/cyan]")
        sys.exit(0)
    else:
        console.print("[red]Invalid choice. Exiting.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
