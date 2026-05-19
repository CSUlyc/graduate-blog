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
					label: '开始',
					items: [
						// Each item here is one entry in the navigation menu.
						{ label: '首页', slug: '' },
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
