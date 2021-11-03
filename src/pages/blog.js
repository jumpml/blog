// import React from "react"
// import Layout from "../components/layout"

// import style from "./index.module.css"

// const IndexPage = () => {
//   return (
//     <Layout>
//       <section className={style.wrapper}>
//         <p>Coming Soon!</p>
//         <p>We will have lots of articles on several interesting topics.</p>
//       </section>
//     </Layout>
//   )
// }

// export default IndexPage

import React from "react"
import { graphql } from "gatsby"
import PostLink from "../components/post-link"
import Layout from "../components/layout"
import SEO from "../components/seo"

import style from "./blog.module.css"

const BlogPage = ({
  data: {
    allMarkdownRemark: { edges },
  },
}) => {
  const Posts = edges
    .filter(edge => !!edge.node.frontmatter.date) // You can filter your posts based on some criteria
    .map(edge => <PostLink key={edge.node.id} post={edge.node} />)
  // return <div>{Posts}</div>

  return (
    <Layout>
      <SEO
        title="JumpML. Learn and build useful things!"
        description="JumpML Website"
        image="/jumpML.svg"
        pathname="/"
        // Boolean indicating whether this is an article:
        // article
      />
      <section className={style.wrapper}>
        <h2>Articles</h2>
        <ul>{Posts}</ul>
      </section>
    </Layout>
  )
}
export default BlogPage
export const pageQuery = graphql`
  query MyQuery {
    allMarkdownRemark(sort: { order: DESC, fields: frontmatter___date }) {
      edges {
        node {
          excerpt
          id
          fields {
            slug
          }
          frontmatter {
            title
            date
            subject
            author
            featimg {
              childImageSharp {
                fluid(maxWidth: 300) {
                  ...GatsbyImageSharpFluid
                }
              }
            }
          }
        }
      }
    }
  }
`
