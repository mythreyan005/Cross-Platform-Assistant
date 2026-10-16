import { render, screen } from '@testing-library/react';
import App from './App';

test('renders KnowledgeAI header and project controls', () => {
  render(<App />);
  const headerElement = screen.getByText(/KnowledgeAI/i);
  expect(headerElement).toBeInTheDocument();

  const newChatBtn = screen.getByText(/New Conversation/i);
  expect(newChatBtn).toBeInTheDocument();

  const dhis2Btn = screen.getByText(/DHIS2 Core/i);
  expect(dhis2Btn).toBeInTheDocument();
});
