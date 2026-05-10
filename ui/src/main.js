import { mount } from 'svelte';
import App from './App.svelte';
import './app.css';

// Why: Svelte 5 uses the `mount` API rather than `new App({ target })`.
// What: Boot the root component into #app.
// Test: `npm run build && npm run preview` renders the sidebar + collections.
const app = mount(App, { target: document.getElementById('app') });

export default app;
