import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({plugins:[react()],server:{proxy:{'/api':'http://127.0.0.1:8501','/image':'http://127.0.0.1:8501','/query':'http://127.0.0.1:8501'}},build:{sourcemap:false}})
