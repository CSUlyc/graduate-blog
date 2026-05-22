// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'Graduate Blog',
			description: 'A blog about my graduate school journey.',
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