// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
	integrations: [
		starlight({
			title: 'Graduate Blog',
			description: 'A blog about my graduate school journey.',
			customCss: ['katex/dist/katex.min.css'],
			pagefind: false,
			components: {
				Search: './src/components/SiteSearch.astro',
			},
			sidebar: [
				{ slug: 'index', label: '首页' },
				{
					label: 'D2L 学习笔记',
					items: [{ autogenerate: { directory: 'project1' } }],
				},
				{
					label: '项目展示',
					items: [{ autogenerate: { directory: 'projects' } }],
				},
			],
		}),
	],
});
