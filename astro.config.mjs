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
				{
					label: 'D2L 学习笔记',
					items: [
						// Each item here is one entry in the navigation menu.
						{ slug: 'index', label: '首页' },
						{ slug: '04-数据操作与预处理', label: '04-数据操作与预处理' },
					],
				},
				{
					label: '项目展示',
					items: [
						{ label: 'D2L / RNN 实验探索', slug: 'projects/d2l-rnn' },
					],
				},
			],
		}),
	],
});
